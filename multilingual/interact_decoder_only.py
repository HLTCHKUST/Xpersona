# Implemented based on https://github.com/huggingface/transfer-learning-conv-ai

import logging
import random
from argparse import ArgumentParser
from itertools import chain
from pprint import pformat
import warnings
import json
import torch
import torch.nn.functional as F

from transformers import Model2Model, BertTokenizer, BertConfig, BertForMaskedLM, BertJapaneseTokenizer
from train_decoder_only import SPECIAL_TOKENS, add_special_tokens_, build_input_from_segments
from utils import get_dataset

LANG_MAP = {"<en>": "En", "<fr>":"Fr", "<it>":"It", "<id>":"Id", "<jp>":"Jp", "<ko>":"Ko", "<zh>":"Zh"}


def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if (
            (cp >= 0x4E00 and cp <= 0x9FFF)
            or (cp >= 0x3400 and cp <= 0x4DBF)  #
            or (cp >= 0x20000 and cp <= 0x2A6DF)  #
            or (cp >= 0x2A700 and cp <= 0x2B73F)  #
            or (cp >= 0x2B740 and cp <= 0x2B81F)  #
            or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
            or (cp >= 0xF900 and cp <= 0xFAFF)
            or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
        ):  #
            return True

        return False


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


def sample_sequence(personality, history, tokenizer, model, args, lang, special_map, current_output=None, selfmem = []):
    special_tokens_ids = [special_map[token] for token in SPECIAL_TOKENS]
    if current_output is None:
        current_output = []
    if args.test_lang in ["En", "Fr", "It", "Id", "Jp", "Ko", "Zh"]:
        lang=None
    for i in range(args.max_length):

        instance = build_input_from_segments(personality, history, current_output, tokenizer, lang, lm_labels=True, with_eos=False)

        input_ids = torch.tensor(instance["input_ids"], device=args.device).unsqueeze(0)
        token_type_ids = torch.tensor(instance["token_type_ids"], device=args.device).unsqueeze(0)
        logits, *_ = model(input_ids = input_ids, token_type_ids= token_type_ids)

        if isinstance(logits, tuple):  # for gpt2 and maybe others
            logits = logits[0]
        for i in selfmem: # for previous response
            logits[0, -1, i] /= args.repetition_penalty
        for i in current_output: # for current response
            logits[0, -1, i] /= args.repetition_penalty
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

    return current_output

def run():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="", help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--dataset_cache", type=str, default='./dataset_cache', help="Path or url of the dataset cache")
    parser.add_argument("--model", type=str, default="bert", help="Model type")  # anything besides gpt2 will load openai-gpt
    parser.add_argument("--model_checkpoint", type=str, default="", help="Path, url or short name of the model")
    parser.add_argument("--max_turns", type=int, default=3, help="Number of previous utterances to keep in history")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument('--repetition_penalty', default=1.3, type=float, help="penalize the repetition")

    parser.add_argument("--no_sample", action='store_true', help="Set to use greedy decoding instead of sampling")
    parser.add_argument("--max_length", type=int, default=30, help="Maximum length of the output utterances")
    parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the output utterances")
    parser.add_argument("--seed", type=int, default=0, help="Seed")
    parser.add_argument("--temperature", type=int, default=0.7, help="Sampling softmax temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
    parser.add_argument("--test_lang", type=str, default="", help="test monolingual model")
    parser.add_argument("--self_play", action='store_true', help="two bot with different persona chat with each other")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__file__)
    logger.info(pformat(args))

    if args.seed != 0:
    	random.seed(args.seed)
    	torch.random.manual_seed(args.seed)
    	torch.cuda.manual_seed(args.seed)


    logger.info("Get pretrained model and tokenizer")
    tokenizer_class, model_class = (BertTokenizer,BertForMaskedLM)
    tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)
    if args.test_lang == "Jp":
        tokenizer = BertJapaneseTokenizer.from_pretrained(args.model_checkpoint)
    bertconfig = BertConfig.from_pretrained(args.model_checkpoint)
    bertconfig.is_decoder=False # for not initailize crossattention
    model = model_class.from_pretrained(args.model_checkpoint, **{"config":bertconfig})
    model.config.is_decoder = True # for causal mask

    # model = model_class.from_pretrained(args.model_checkpoint)
    # model.config.is_decoder = True
    with open(args.model_checkpoint+"/added_tokens.json", encoding="utf-8") as f:
        special_map = json.load(f)

    model.load_state_dict(torch.load(args.model_checkpoint+"/pytorch_model.bin"))
    model.to(args.device)
    model.eval()
    

    
    lang = input("choose one language from : en, fr, it, id, jp, ko, zh\n")
    while lang not in ["en", "fr", "it", "id", "jp", "ko", "zh"]:
        print('Choose correct language!')
        lang = input("choose one language from : en, fr, it, id, jp, ko, zh\n")
    lang = "<{}>".format(lang)

    logger.info("Sample a personality of {}".format(lang))

    dataset = get_dataset(tokenizer, args.dataset_path, args.dataset_cache, args.test_lang)

    personalities = [dialog["persona"] for dialog in dataset["test"][LANG_MAP[lang]]]

    history = []
    dont_repeatself = [[],[]] #persona1 and persona2, use for selfplay
    save_path = "selfplay/"
    if args.test_lang in ["En", "Fr", "It", "Id", "Jp", "Ko", "Zh"]:
        prefix = "mono_"
    else:
        prefix = "multi_"
    if args.self_play:
        for j in range(50): # how many conversations?
            print("===================================================")
            personality_1 = random.choice(personalities)
            logger.info("personality of bot 1: %s", tokenizer.decode(chain(*personality_1)))
            personality_2 = random.choice(personalities)
            logger.info("personality of bot 2: %s", tokenizer.decode(chain(*personality_2)))

            starters = {"<en>":["hello, how are you doing today?", "hi, how are you?", "hello , what are you doing ?"], 
                        "<zh>":["你好，今天在做什么？", "嗨，你好吗？","你好，你今天怎么样 ？"], 
                        "<it>":["Ciao, come va oggi?", "Ciao, come stai?", "Ciao, cosa stai facendo ?"],
                        "<jp>":["こんにちは、今日はどうですか？","こんにちは、元気ですか？","やあ、元気 ？"],
                        "<ko>":["안녕, 오늘 어떻게 지내니?","안녕하세요?","안녕, 너는 무엇을 하고 있니?"],
                        "<id>":["Hai apa kabarmu hari ini?", "Hai apa kabar?", "Halo apa yang kamu lakukan ?"],
                        "<fr>":["Bonjour comment allez-vous aujourd'hui?","salut comment ca va?","salut que fais tu ?"]
                        }

            starter = random.choice(starters[lang])

            print(starter)
            self_conversations = {"dialog":[{"speaker":"human_evaluator","text":starter}]}
            history.append(tokenizer.encode(starter))
            
            for i in range(13):
                with torch.no_grad():
                    out_ids = sample_sequence(personality_1 if i%2 else personality_2, history, tokenizer, model, args, lang, special_map, selfmem = dont_repeatself[i%2])
                dont_repeatself[i%2] = out_ids #only remember last turn
                history.append(out_ids)
                history = history[-args.max_turns:]
                out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
                if lang in ["<jp>", "<zh>"]:
                    print("".join(out_text.split()))
                    self_conversations["dialog"].append({"speaker":"human_evaluator" if i%2 else "model","text":"".join(out_text.split())})
                else:
                    print(out_text)
                    self_conversations["dialog"].append({"speaker":"human_evaluator" if i%2 else "model","text":out_text})
            with open(save_path+prefix+LANG_MAP[lang]+'_output_new.jsonl', 'a') as outfile:
                json.dump(self_conversations, outfile)
                outfile.write('\n')
    else:
        personality = random.choice(personalities)
        logger.info("Selected personality: %s", tokenizer.decode(chain(*personality)))

        while True:
            raw_text = input(">>> ")
            while not raw_text:
                print('Prompt should not be empty!')
                raw_text = input(">>> ")
            history.append(tokenizer.encode(raw_text))
            with torch.no_grad():
                out_ids = sample_sequence(personality, history, tokenizer, model, args, lang, special_map)
            history.append(out_ids)
            history = history[-args.max_turns:]
            out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
            if lang in ["<jp>", "<zh>"]:
                print("".join(out_text.split()))
            else:
                print(out_text)


if __name__ == "__main__":
    run()
