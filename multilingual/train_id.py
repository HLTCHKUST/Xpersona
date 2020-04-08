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
MODEL_INPUTS = ["encoder_mask", "decoder_mask", "encoder_input_ids", "decoder_input_ids", "lm_labels", "token_type_ids", "decoder_lang_id"]
LANG_2_MODEL = {"En":"bert-base-cased", "Zh":"bert-base-chinese", "Jp":"bert-base-japanese-whole-word-masking", "It":"dbmdz/bert-base-italian-cased"}

#PADDED_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]

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

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        item = self.data[index]
        return item
    def __len__(self):
        return self.dataset_len

def collate_fn(data):
    # batch padding
    padding = 0 # [PAD] is 0
    max_input = max(len(x["input_ids"]) for x in data) #max input len
    max_response = max(len(x["lm_labels"]) for x in data)
    #print(max_input, max_response)
    padded_dataset = {"encoder_mask":[],"decoder_mask":[], "encoder_input_ids":[],"decoder_input_ids":[], "lm_labels":[], "token_type_ids":[], "decoder_lang_id":[]}
    for x in data:
        padded_dataset["encoder_mask"].append(len(x["input_ids"])*[1] + [0]*(max_input-len(x["input_ids"])))
        padded_dataset["encoder_input_ids"].append(x["input_ids"]+ [padding]*(max_input-len(x["input_ids"])) )
        padded_dataset["token_type_ids"].append( x["token_type_ids"]+ [padding]*(max_input-len(x["input_ids"])) )
        padded_dataset["lm_labels"].append( x["lm_labels"]+ [-1]*(max_response-len(x["lm_labels"]))  )
        padded_dataset["decoder_input_ids"].append(x["lm_labels"]+ [padding]*(max_response-len(x["lm_labels"])))
        padded_dataset["decoder_mask"].append(len(x["lm_labels"])*[1] + [0]*(max_response-len(x["lm_labels"])) )
        padded_dataset["decoder_lang_id"].append([x["lang_id"]])

    for input_name in MODEL_INPUTS:
        padded_dataset[input_name] = torch.tensor(padded_dataset[input_name])
    return padded_dataset



def add_special_tokens_(model, tokenizer, update_model=True):
    """ Add special tokens to the tokenizer and the model if they have not already been added. """
    orig_num_tokens = len(tokenizer)
    num_added_tokens = tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN) # doesn't add if they are already there
    #print("coab::",len(tokenizer.vocab))
    if (num_added_tokens > 0 and update_model):
        model.encoder.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)
        model.decoder.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)
    #print(model.encoder.embeddings.word_embeddings.weight.shape)
    #print(model.decoder.bert.embeddings.word_embeddings.weight.shape)

def build_input_from_segments(persona, history, reply, tokenizer, lang_id=None, lm_labels=False, with_eos=True):
    """ Build a sequence of input from 3 segments: persona, history and last reply. """
    bos, eos, persona_token, speaker1, speaker2 = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:5])
    if lang_id is None:
        lang_id_token = 0
    else:
        lang_id_token = tokenizer.convert_tokens_to_ids(lang_id)
    personality = []
    for sent in persona:
        personality+=[persona_token]+sent
    sequence = [personality] + history  #+ [reply + ([eos] if with_eos else [])]
    sequence = [sequence[0]] + [[speaker2 if i % 2 else speaker1] + s for i, s in enumerate(sequence[1:])]
    response = [bos] + reply + ([eos] if with_eos else [])
    instance = {}
    instance["input_ids"] = list(chain(*sequence))
    instance["token_type_ids"] =  [persona_token]*len(sequence[0]) + [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence[1:]) for _ in s]
    instance["lm_labels"] = [-1] * len(response)
    instance["lang_id"] = lang_id_token

    if lm_labels:
        instance["lm_labels"] = response
    return instance


def get_data_loaders(args, tokenizer):
    """ Prepare the dataset for training and evaluation """
    personachat = get_dataset(tokenizer, args.dataset_path, args.dataset_cache, args.train_lang)
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
                    instance = build_input_from_segments(dial["persona"], dial["history"][-args.max_turns:], dial["response"], tokenizer, lang_id="<{}>".format(lang.lower()),  lm_labels = True)
                    datasets[dataset_name].append(instance)  #all langs together


    logger.info("Build train and validation dataloaders")
    train_dataset = DatasetTrain(datasets["train"])
    valid_dataset = DatasetTrain(datasets["valid"])

    #logger.info("Build train and validation dataloaders")
    #train_dataset, valid_dataset = TensorDataset(*tensor_datasets["train"]), TensorDataset(*tensor_datasets["valid"])
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset) if args.distributed else None
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, shuffle=(not args.distributed), collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=args.valid_batch_size, shuffle=False, collate_fn=collate_fn)

    # logger.info("Train dataset (Batch, Candidates, Seq length): {}".format(train_dataset.tensors[0].shape))
    # #logger.info("Train dataset (Batch, Candidates, Seq length): {}".format(train_dataset.tensors[1].shape))
    # logger.info("Valid dataset (Batch, Candidates, Seq length): {}".format(valid_dataset.tensors[0].shape))
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
    parser.add_argument("--random_init",  action='store_true', help="If true random initailze the model")
    parser.add_argument("--train_lang", type=str, default="", help="train monolingual model, defaul: multilingual model")
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
    model = Model2Model.from_pretrained(model_path)

    # tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    # if args.random_init:
    #     config = BertConfig.from_pretrained('bert-base-multilingual-cased')
    #     config.is_decoder = True
    #     bert_decoder = BertForMaskedLM(config)
    #     model = Model2Model.from_pretrained('bert-base-multilingual-cased', decoder_model=bert_decoder)
    # else:
    #     model = Model2Model.from_pretrained('bert-base-multilingual-cased')
    #     model_dict = model.state_dict()
    #     # initialize crossattention with selfattention
    #     model_dict.update({ name: model_dict[name.replace("crossattention", "attention")] for name in model_dict if "crossattention" in name })
    #     model.load_state_dict(model_dict)
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
        
        #batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
        encoder_mask, decoder_mask, encoder_input_ids, decoder_input_ids, lm_labels, token_type_ids, decoder_lang_id = batch
        model_kwargs = {"encoder_token_type_ids":token_type_ids, "decoder_token_type_ids":decoder_lang_id, "encoder_attention_mask":encoder_mask, "decoder_attention_mask":decoder_mask, "decoder_lm_labels":lm_labels}
        lm_loss, prediction_scores, *_ = model(encoder_input_ids = encoder_input_ids, decoder_input_ids =decoder_input_ids, **model_kwargs)

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
            #batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
            encoder_mask, decoder_mask, encoder_input_ids, decoder_input_ids, lm_labels, token_type_ids, decoder_lang_id = batch
            logger.info(tokenizer.decode(encoder_input_ids[0, :].tolist()))
            # if we dont send labels to model, it doesnt return losses
            model_kwargs = {"encoder_token_type_ids":token_type_ids, "decoder_token_type_ids":decoder_lang_id, "encoder_attention_mask":encoder_mask, "decoder_attention_mask":decoder_mask}

            lm_logits, *_ = model(encoder_input_ids = encoder_input_ids, decoder_input_ids =decoder_input_ids, **model_kwargs)
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
        log_dir +="_lang_id"
        if args.random_init:
            log_dir = log_dir + "_random_init"
        tb_logger = TensorboardLogger(log_dir)

        tb_logger.attach(trainer, log_handler=OutputHandler(tag="training", metric_names=["loss"]), event_name=Events.ITERATION_COMPLETED)
        tb_logger.attach(trainer, log_handler=OptimizerParamsHandler(optimizer), event_name=Events.ITERATION_STARTED)
        tb_logger.attach(evaluator, log_handler=OutputHandler(tag="validation", metric_names=list(metrics.keys()), another_engine=trainer), event_name=Events.EPOCH_COMPLETED)

        checkpoint_handler = ModelCheckpoint(log_dir, 'checkpoint', save_interval=1, n_saved=3)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {'mymodel': getattr(model, 'module', model)})  # "getattr" takes care of distributed encapsulation

        torch.save(args, log_dir + '/model_training_args.bin')
        if args.distributed:
            getattr(model.module, 'encoder', model).config.to_json_file(os.path.join(log_dir, CONFIG_NAME)) # the config for encoder and decoder should be the same
        else:
            getattr(model, 'encoder', model).config.to_json_file(os.path.join(log_dir, CONFIG_NAME)) # the config for encoder and decoder should be the same
        tokenizer.save_pretrained(log_dir)

    # Run the training
    trainer.run(train_loader, max_epochs=args.n_epochs)

    # On the main process: close tensorboard logger and rename the last checkpoint (for easy re-loading with OpenAIGPTModel.from_pretrained method)
    if args.local_rank in [-1, 0] and args.n_epochs > 0:
        os.rename(checkpoint_handler._saved[-1][1][-1], os.path.join(log_dir, WEIGHTS_NAME))  # TODO: PR in ignite to have better access to saved file paths (cleaner)
        tb_logger.close()

if __name__ == "__main__":
    train()