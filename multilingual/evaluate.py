import logging
import random
import numpy as np
import math
from argparse import ArgumentParser
from itertools import chain
from pprint import pformat
import warnings
import json
import torch
import torch.nn.functional as F
from transformers import Model2Model, BertTokenizer, BertConfig, BertJapaneseTokenizer
from train_id import SPECIAL_TOKENS, add_special_tokens_
from utils import get_dataset
import os

from metrics.eval_metrics import moses_multi_bleu

LOSS = torch.nn.CrossEntropyLoss(ignore_index=-1)


def build_input_from_segments(persona, history, reply, tokenizer, lang_id, special_map, lm_labels=False, with_eos=True):
    """ Build a sequence of input from 3 segments: persona, history and last reply. """
    bos, eos, persona_token, speaker1, speaker2 = [special_map[token] for token in SPECIAL_TOKENS[:5]]
    if lang_id is None:
        lang_id_token = 0
    else:
        lang_id_token = [special_map[lang_id]]
    
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

    # print(tokenizer.decode(instance["lang_id"]))
    # print(tokenizer.decode(instance["input_ids"]))
    # print(tokenizer.decode(instance["token_type_ids"]) )
    # print(tokenizer.decode(response))
    return instance


def top_filtering(logits, top_k=0., top_p=0.9, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits


def sample_sequence(personality, history, tokenizer, model, args, lang, special_map, current_output=None, ref = None, ):
    special_tokens_ids = [special_map[token] for token in SPECIAL_TOKENS]

    if current_output is None:
        current_output = []
    if args.test_lang in ["En", "Fr", "It", "Id", "Jp", "Ko", "Zh"]:
        lang=None
    for i in range(args.max_length):
        instance = build_input_from_segments(personality, history, current_output, tokenizer, lang, special_map, lm_labels=True, with_eos=False)

        input_ids = torch.tensor(instance["input_ids"], device=args.device).unsqueeze(0)
        token_type_ids = torch.tensor(instance["token_type_ids"], device=args.device).unsqueeze(0)
        encoder_mask = torch.tensor(len(instance["input_ids"])*[1], device=args.device).unsqueeze(0)
        decoder_mask = torch.tensor(len(instance["lm_labels"])*[1], device=args.device).unsqueeze(0)
        decoder_type_ids = torch.tensor(instance["lang_id"], device=args.device).unsqueeze(0)
        #print(decoder_type_ids)

        model_kwargs = {"encoder_token_type_ids":token_type_ids,"decoder_token_type_ids":decoder_type_ids, "encoder_attention_mask":encoder_mask, "decoder_attention_mask":decoder_mask}
        decoder_input_ids = torch.tensor(instance["lm_labels"], device=args.device).unsqueeze(0)
        
        logits, *_ = model(encoder_input_ids = input_ids, decoder_input_ids = decoder_input_ids, **model_kwargs)

        if isinstance(logits, tuple):  # for gpt2 and maybe others
            logits = logits[0]
        logits = logits[0, -1, :] / args.temperature
        logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
        probs = F.softmax(logits, dim=-1)

        prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)
        if i < args.min_length and prev.item() in special_tokens_ids:
            while prev.item() in special_tokens_ids:
                if probs.max().item() == 1:
                    warnings.warn("Warning: model generating special token with probability 1.")
                    break  # avoid infinitely looping over special token
                prev = torch.multinomial(probs, num_samples=1)

        if prev.item() in special_tokens_ids:
            break
        current_output.append(prev.item())

    # compute PPL
    instance = build_input_from_segments(personality, history, ref, tokenizer, lang, special_map, lm_labels=True, with_eos=True)
    decoder_mask = torch.tensor(len(instance["lm_labels"])*[1], device=args.device).unsqueeze(0)
    model_kwargs = {"encoder_token_type_ids":token_type_ids,"decoder_token_type_ids":decoder_type_ids, "encoder_attention_mask":encoder_mask, "decoder_attention_mask":decoder_mask}
    decoder_input_ids = torch.tensor(instance["lm_labels"], device=args.device).unsqueeze(0)
    logits, *_ = model(encoder_input_ids = input_ids, decoder_input_ids = decoder_input_ids, **model_kwargs)
    lm_logits_flat_shifted = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
    lm_labels_flat_shifted = decoder_input_ids[..., 1:].contiguous().view(-1)
    loss = LOSS(lm_logits_flat_shifted, lm_labels_flat_shifted)
    return current_output, loss.item()

def run():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="", help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--dataset_cache", type=str, default='./dataset_cache', help="Path or url of the dataset cache")
    parser.add_argument("--model", type=str, default="multi-bert", help="Model type")  # anything besides gpt2 will load openai-gpt
    parser.add_argument("--model_checkpoint", type=str, default="", help="Path, url or short name of the model")
    parser.add_argument("--max_turns", type=int, default=3, help="Number of previous utterances to keep in history")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")

    parser.add_argument("--no_sample", action='store_true', help="Set to use greedy decoding instead of sampling")
    parser.add_argument("--max_length", type=int, default=20, help="Maximum length of the output utterances")
    parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the output utterances")
    parser.add_argument("--seed", type=int, default=0, help="Seed")
    parser.add_argument("--temperature", type=int, default=0.7, help="Sampling softmax temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
    parser.add_argument("--test_lang", type=str, default="", help="test monolingual model")
    parser.add_argument("--no_lang_id", action='store_true')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__file__)
    logger.info(pformat(args))

    if args.seed != 0:
    	random.seed(args.seed)
    	torch.random.manual_seed(args.seed)
    	torch.cuda.manual_seed(args.seed)


    logger.info("Get pretrained model and tokenizer")
    tokenizer = BertTokenizer.from_pretrained(args.model_checkpoint)
    if args.test_lang == "Jp":
        tokenizer = BertJapaneseTokenizer.from_pretrained(args.model_checkpoint)
    bertconfig = BertConfig.from_pretrained(args.model_checkpoint)
    bertconfig.is_decoder=True
    model = Model2Model.from_pretrained(args.model_checkpoint, **{"decoder_config":bertconfig})

    with open(args.model_checkpoint+"/added_tokens.json", encoding="utf-8") as f:
        special_map = json.load(f)

    model.load_state_dict(torch.load(args.model_checkpoint+"/pytorch_model.bin"))
    model.to(args.device)
    model.eval()
    
    personachat = get_dataset(tokenizer, args.dataset_path, args.dataset_cache, args.test_lang)
    persona_text = {}
    context_text = {}
    output_text = {}
    ref_text = {}
    ppl = {}
    BLEU_score = {}
    if args.test_lang in ["En", "Fr", "It", "Id", "Jp", "Ko", "Zh"]:
        logdir = args.test_lang+"_bert2bert/"
    else:
        logdir = "multilingual_bert2bert/"
    for lang, dials in personachat["test"].items():
        if args.test_lang in ["En", "Fr", "It", "Id", "Jp", "Ko", "Zh"]:
            if lang!=args.test_lang:
                continue
        persona_text[lang] = []
        context_text[lang] = []
        output_text[lang] = []
        ref_text[lang] = []
        loss_list = []
        for dial in dials: #dial: {"persona":[], "history":[], "response":str}
            with torch.no_grad():
                out_ids, loss = sample_sequence(dial["persona"], dial["history"][-args.max_turns:], tokenizer, model, args, "<{}>".format(lang.lower()), special_map, ref = dial["response"])
                output_text[lang].append(tokenizer.decode(out_ids, skip_special_tokens=True))
                # print(tokenizer.decode(dial["history"][-1]))
                # print(output_text[lang][-1])
                # print("-")
                ref_text[lang].append(tokenizer.decode(dial["response"], skip_special_tokens=True))
                context_text[lang].append([tokenizer.decode(turn, skip_special_tokens=True) for turn in dial["history"]])
                persona_text[lang].append([tokenizer.decode(sent, skip_special_tokens=True) for sent in dial["persona"]])
                loss_list.append(loss)
        ppl[lang] = math.exp(np.mean(loss_list))
        print("{} PPL:{}".format(lang, ppl[lang]))
        if not os.path.exists("results/"+logdir):
            os.makedirs("results/"+logdir)
        with open("results/"+logdir+lang+"_output.txt", 'w', encoding='utf-8') as f:
            for line in output_text[lang]:
                f.write(line)
                f.write('\n')

        with open("results/"+logdir+lang+"_ref.txt", 'w', encoding='utf-8') as f:
            for line in ref_text[lang]:
                f.write(line)
                f.write('\n')
        print("Computing BLEU for {}".format(lang))
        BLEU = moses_multi_bleu(np.array(output_text[lang]),np.array(ref_text[lang]))
        print("BLEU:{}".format(BLEU))
        BLEU_score[lang] = BLEU

        with open("results/"+logdir+lang+"_all.txt", 'w', encoding='utf-8') as f:
            for i in range(len(ref_text[lang])):
                f.write("=====================================================\n")
                f.write("Persona:\n")
                for sent in persona_text[lang][i]:
                    f.write("".join(sent.split()) if lang in ["jp", "zh"] else sent )
                    f.write('\n')
                f.write("History:\n")
                for sent in context_text[lang][i]:
                    f.write("".join(sent.split()) if lang in ["jp", "zh"] else sent)
                    f.write('\n')
                f.write("Response: ")
                f.write("".join(output_text[lang][i].split()) if lang in ["jp", "zh"] else output_text[lang][i])
                f.write('\n')
                f.write("Ref: ")
                f.write("".join(ref_text[lang][i].split()) if lang in ["jp", "zh"] else ref_text[lang][i])
                f.write('\n')

    with open("results/"+logdir+"BLEU_score.txt", "w", encoding='utf-8') as f:
        for language, score in BLEU_score.items():
            f.write("{}\t PPL:{}, BLEU:{}\n".format(language, ppl[language],score))



if __name__ == "__main__":
    run()