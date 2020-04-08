# Implemented based on https://github.com/huggingface/transfer-learning-conv-ai

import os
import math
import logging
from pprint import pformat
from argparse import ArgumentParser
from collections import defaultdict
from itertools import chain

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset, Dataset
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Accuracy, Loss, MetricsLambda, RunningAverage
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler
from transformers import (AdamW, Model2Model, BertTokenizer, WEIGHTS_NAME, CONFIG_NAME, PreTrainedEncoderDecoder, BertConfig, BertModel, BertForMaskedLM, BertJapaneseTokenizer)

from utils import get_dataset, make_logdir



SPECIAL_TOKENS = [ "<bos>", "<eos>", "<persona>", "<speaker1>", "<speaker2>","<en>", "<fr>", "<it>", "<id>", "<jp>", "<ko>", "<zh>", "<pad>"]
ATTR_TO_SPECIAL_TOKEN = {'bos_token': '<bos>', 'eos_token': '<eos>', 'pad_token': '<pad>',
                         'additional_special_tokens': ("<persona>", '<speaker1>', '<speaker2>',"<en>", "<fr>", "<it>", "<id>", "<jp>", "<ko>", "<zh>")}
MODEL_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]
LANG_2_MODEL = {"En":"bert-base-cased", "Zh":"bert-base-chinese", "Jp":"bert-base-japanese-whole-word-masking", "It":"dbmdz/bert-base-italian-cased"}
logger = logging.getLogger(__file__)

def average_distributed_scalar(scalar, args):
    """ Average a scalar over the nodes if we are in distributed training. We use this for distributed evaluation. """
    if args.local_rank == -1:
        return scalar
    scalar_t = torch.tensor(scalar, dtype=torch.float, device=args.device) / torch.distributed.get_world_size()
    torch.distributed.all_reduce(scalar_t, op=torch.distributed.ReduceOp.SUM)
    return scalar_t.item()


class DatasetTrain(Dataset):
    """Custom data.Dataset compatible with DataLoader."""
    def __init__(self, data):
        self.data = data
        self.dataset_len = len(self.data)
        self.max_len = max(len(x["input_ids"]) for x in self.data) 
    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        item = self.data[index]
        return item
    def __len__(self):
        return self.dataset_len

def collate_fn(data):
    # batch padding
    padding = 0
    max_input = max(len(x["input_ids"]) for x in data) #max input len
    padded_dataset = {"input_ids":[], "lm_labels":[], "token_type_ids":[]}
    for x in data:
        padded_dataset["token_type_ids"].append( x["token_type_ids"]+ [padding]*(max_input-len(x["input_ids"])) )
        padded_dataset["lm_labels"].append( x["lm_labels"]+ [-1]*(max_input-len(x["lm_labels"]))  )
        padded_dataset["input_ids"].append(x["input_ids"]+ [padding]*(max_input-len(x["input_ids"])))
    for input_name in MODEL_INPUTS:
        padded_dataset[input_name] = torch.tensor(padded_dataset[input_name])
    return padded_dataset


def add_special_tokens_(model, tokenizer, update_model=True):
    """ Add special tokens to the tokenizer and the model if they have not already been added. """
    orig_num_tokens = len(tokenizer)
    num_added_tokens = tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN) # doesn't add if they are already there
    #print("coab::",len(tokenizer.vocab))
    if (num_added_tokens > 0 and update_model):
        model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)


def build_input_from_segments(persona, history, reply, tokenizer, lang_id=None, lm_labels=False, with_eos=True):
    """ Build a sequence of input from 3 segments: persona, history and last reply. """
    bos, eos, persona_token, speaker1, speaker2 = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:5])
    if lang_id is None:
        lang_id_token = bos
    else:
        lang_id_token = tokenizer.convert_tokens_to_ids(lang_id)
    personality = []
    for sent in persona:
        personality+=[persona_token]+sent
    sequence = [personality] + history
    sequence = [sequence[0]] + [[speaker2 if i % 2 else speaker1] + s for i, s in enumerate(sequence[1:])]
    response = [lang_id_token] + reply + ([eos] if with_eos else [])
    sequence += [response]
    instance = {}
    instance["input_ids"] = list(chain(*sequence))
    instance["token_type_ids"] =  [persona_token]*len(sequence[0]) + [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence[1:-1]) for _ in s] + [lang_id_token]*len(response)
    instance["lm_labels"] = [-1] * len(instance["input_ids"])

    # print(tokenizer.decode(instance["input_ids"]))
    # print(tokenizer.decode(instance["token_type_ids"]) )
    if lm_labels:
        instance["lm_labels"] = ([-1] * sum(len(s) for s in sequence[:-1])) + [-1] + sequence[-1][1:]
        # print(instance["input_ids"])
        # print(instance["lm_labels"])
    return instance


def get_data_loaders(args, tokenizer):
    """ Prepare the dataset for training and evaluation """
    personachat = get_dataset(tokenizer, args.dataset_path, args.dataset_cache, args.train_lang)
    #personachat = get_dataset(tokenizer, args.dataset_path, args.dataset_cache)
    _ = personachat.pop("test", None)
    logger.info("Build inputs and labels")
    datasets = {"train": [], "valid": []}

    if args.train_lang in ["En", "Fr", "It", "Id", "Jp", "Ko", "Zh"]: #monolingual data
        for dataset_name, dataset in personachat.items():
            for dial in dataset[args.train_lang]: #dial: {"persona":[], "history":[], "response":str}
                instance = build_input_from_segments(dial["persona"], dial["history"][-args.max_turns:], dial["response"], tokenizer, lm_labels = True)
                datasets[dataset_name].append(instance)  
    else: #multilingual data
        for dataset_name, dataset in personachat.items():
            for lang, dials in dataset.items():
                for dial in dials: #dial: {"persona":[], "history":[], "response":str}
                    instance = build_input_from_segments(dial["persona"], dial["history"][-args.max_turns:], dial["response"], tokenizer, lang_id="<{}>".format(lang.lower()) if not args.no_lang_id else None,  lm_labels = True)
                    if len(instance["input_ids"])<400:
                        datasets[dataset_name].append(instance)  #all langs together

    logger.info("Build train and validation dataloaders")
    train_dataset = DatasetTrain(datasets["train"])
    valid_dataset = DatasetTrain(datasets["valid"])
    print(train_dataset.max_len, valid_dataset.max_len)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset) if args.distributed else None
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, shuffle=(not args.distributed), collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=args.valid_batch_size, shuffle=False, collate_fn=collate_fn)

    logger.info("Train dataset length: {}".format(len(train_dataset)))
    logger.info("Valid dataset length: {}".format(len(valid_dataset)))
    return train_loader, valid_loader, train_sampler, valid_sampler


def train():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="", help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--dataset_cache", type=str, default='./dataset_cache', help="Path or url of the dataset cache")
    parser.add_argument("--model_checkpoint", type=str, default="multi-bert", help="Path, url or short name of the model")
    parser.add_argument("--num_candidates", type=int, default=2, help="Number of candidates for training")
    parser.add_argument("--max_turns", type=int, default=3, help="Number of previous turns to keep in history")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=4, help="Batch size for validation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Accumulate gradients on several steps")
    parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate")
    parser.add_argument("--lm_coef", type=float, default=1.0, help="LM loss coefficient")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--n_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--personality_permutations", type=int, default=1, help="Number of permutations of personality sentences")
    parser.add_argument("--eval_before_start", action='store_true', help="If true start with a first evaluation before training")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--fp16", type=str, default="", help="Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training (-1: not distributed)")
    parser.add_argument("--train_lang", type=str, default="", help="train monolingual model, defaul: multilingual model")
    parser.add_argument("--no_lang_id", action='store_true', help="no language id as input")
    args = parser.parse_args()

    # logging is set to INFO (resp. WARN) for main (resp. auxiliary) process. logger.info => log main process only, logger.warning => log all processes
    logging.basicConfig(level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Running process %d", args.local_rank)  # This is a logger.warning: it will be printed by all distributed processes
    logger.info("Arguments: %s", pformat(args))

    # Initialize distributed training if needed
    args.distributed = (args.local_rank != -1)
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    # Model
    logger.info("Prepare tokenizer, pretrained model and optimizer.")
    model_path = 'bert-base-multilingual-cased'
    if args.train_lang in ["En", "It", "Jp", "Zh"]: # for Fr Ko Id we use MBERT
        model_path = LANG_2_MODEL[args.train_lang]

    tokenizer = BertTokenizer.from_pretrained(model_path)
    if args.train_lang == "Jp":
        tokenizer = BertJapaneseTokenizer.from_pretrained(model_path)

    bertconfig = BertConfig.from_pretrained(model_path)
    bertconfig.is_decoder=False # for not initailize crossattention
    model = BertForMaskedLM.from_pretrained(model_path, **{"config":bertconfig})
    model.config.is_decoder = True
    model.to(args.device)

    # Add special tokens if they are not already added
    add_special_tokens_(model, tokenizer)
    optimizer = AdamW(model.parameters(), lr=args.lr, correct_bias=True)

    # Prepare model for FP16 and distributed training if needed (order is important, distributed should be the last)
    if args.fp16:
        from apex import amp  # Apex is only required if we use fp16 training
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16)
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    
    logger.info("Prepare datasets")
    train_loader, val_loader, train_sampler, valid_sampler = get_data_loaders(args, tokenizer)

    # Training function and trainer
    def update(engine, batch):
        model.train()
        batch = tuple(batch[input_name].to(args.device) for input_name in MODEL_INPUTS)
        input_ids, lm_labels, token_type_ids = batch
        lm_loss, prediction_scores, *_ = model(input_ids = input_ids, token_type_ids= token_type_ids, lm_labels = lm_labels)
        #batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
        loss = (lm_loss) / args.gradient_accumulation_steps
        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_norm)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        if engine.state.iteration % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        return loss.item()
    trainer = Engine(update)

    # Evaluation function and evaluator (evaluator output is the input of the metrics)
    def inference(engine, batch):
        model.eval()
        with torch.no_grad():
            batch = tuple(batch[input_name].to(args.device) for input_name in MODEL_INPUTS)
            input_ids, lm_labels, token_type_ids = batch
            logger.info(tokenizer.decode(input_ids[0, :].tolist()))
            # if we dont send labels to model, it doesnt return losses
            lm_logits, *_ = model(input_ids = input_ids, token_type_ids= token_type_ids)
            lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
            lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)
            return (lm_logits_flat_shifted, ), (lm_labels_flat_shifted, )
    evaluator = Engine(inference)

    # Attach evaluation to trainer: we evaluate when we start the training and at the end of each epoch
    trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: evaluator.run(val_loader))
    if args.n_epochs < 1:
        trainer.add_event_handler(Events.COMPLETED, lambda _: evaluator.run(val_loader))
    if args.eval_before_start:
        trainer.add_event_handler(Events.STARTED, lambda _: evaluator.run(val_loader))

    # Make sure distributed data samplers split the dataset nicely between the distributed processes
    if args.distributed:
        trainer.add_event_handler(Events.EPOCH_STARTED, lambda engine: train_sampler.set_epoch(engine.state.epoch))
        evaluator.add_event_handler(Events.EPOCH_STARTED, lambda engine: valid_sampler.set_epoch(engine.state.epoch))

    # Linearly decrease the learning rate from lr to zero
    scheduler = PiecewiseLinear(optimizer, "lr", [(0, args.lr), (args.n_epochs * len(train_loader), 0.0)])
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)


    # Prepare metrics - note how we compute distributed metrics
    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")
    metrics = {"nll": Loss(torch.nn.CrossEntropyLoss(ignore_index=-1), output_transform=lambda x: (x[0][0], x[1][0]))}
    metrics.update({"average_nll": MetricsLambda(average_distributed_scalar, metrics["nll"], args)})
    metrics["average_ppl"] = MetricsLambda(math.exp, metrics["average_nll"])
    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    # On the main process: add progress bar, tensorboard, checkpoints and save model, configuration and tokenizer before we start to train
    if args.local_rank in [-1, 0]:
        pbar = ProgressBar(persist=True)
        pbar.attach(trainer, metric_names=["loss"])
        evaluator.add_event_handler(Events.COMPLETED, lambda _: pbar.log_message("Validation: %s" % pformat(evaluator.state.metrics)))

        log_dir = make_logdir(args.model_checkpoint)
        log_dir +="_decoder_only"
        if args.no_lang_id:
            log_dir +="_noid"
        if args.train_lang in ["En", "Fr", "It", "Id", "Jp", "Ko", "Zh"]:
            log_dir += "_"+args.train_lang
        tb_logger = TensorboardLogger(log_dir)

        tb_logger.attach(trainer, log_handler=OutputHandler(tag="training", metric_names=["loss"]), event_name=Events.ITERATION_COMPLETED)
        tb_logger.attach(trainer, log_handler=OptimizerParamsHandler(optimizer), event_name=Events.ITERATION_STARTED)
        tb_logger.attach(evaluator, log_handler=OutputHandler(tag="validation", metric_names=list(metrics.keys()), another_engine=trainer), event_name=Events.EPOCH_COMPLETED)

        checkpoint_handler = ModelCheckpoint(log_dir, 'checkpoint', save_interval=1, n_saved=3)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {'mymodel': getattr(model, 'module', model)})  # "getattr" takes care of distributed encapsulation

        torch.save(args, log_dir + '/model_training_args.bin')
        getattr(model, 'module', model).config.to_json_file(os.path.join(log_dir, CONFIG_NAME)) # the config for encoder and decoder should be the same
        tokenizer.save_pretrained(log_dir)

    # Run the training
    trainer.run(train_loader, max_epochs=args.n_epochs)

    # On the main process: close tensorboard logger and rename the last checkpoint (for easy re-loading with OpenAIGPTModel.from_pretrained method)
    if args.local_rank in [-1, 0] and args.n_epochs > 0:
        os.rename(checkpoint_handler._saved[-1][1][-1], os.path.join(log_dir, WEIGHTS_NAME))  # TODO: PR in ignite to have better access to saved file paths (cleaner)
        tb_logger.close()

if __name__ == "__main__":
    train()