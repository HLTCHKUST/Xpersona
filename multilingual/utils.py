# Implemented based on https://github.com/huggingface/transfer-learning-conv-ai

from datetime import datetime
import json
import logging
import os
import tarfile
import tempfile
import socket

import torch

from transformers import cached_path

from google.cloud import storage, translate
from google.api_core.exceptions import GoogleAPICallError

from translation import collection_batch_translate

logger = logging.getLogger(__file__)

PERSONACHAT_URL = "https://s3.amazonaws.com/datasets.huggingface.co/personachat/personachat_self_original.json"
HF_FINETUNED_MODEL = "https://s3.amazonaws.com/models.huggingface.co/transfer-learning-chatbot/gpt_personachat_cache.tar.gz"

google_lang_id_map = {'en':'en', 'fr':'fr', 'it':'it','id':'id','jp':'ja','ko':'ko','zh':'zh'}

def download_pretrained_model():
    """ Download and extract finetuned model from S3 """
    resolved_archive_file = cached_path(HF_FINETUNED_MODEL)
    tempdir = tempfile.mkdtemp()
    logger.info("extracting archive file {} to temp dir {}".format(resolved_archive_file, tempdir))
    with tarfile.open(resolved_archive_file, 'r:gz') as archive:
        archive.extractall(tempdir)
    return tempdir

def get_google_language_id(lang):
    return google_lang_id_map[lang.lower()]

def get_translated_test_dataset(tokenizer, dataset_path, dataset_cache, google_project_id):
    """ Get tokenized PERSONACHAT dataset."""
    dataset_cache = dataset_cache + '_' + type(tokenizer).__name__
    if dataset_cache and os.path.isfile(dataset_cache):
        logger.info("Load tokenized dataset from cache at %s", dataset_cache)
        dataset = torch.load(dataset_cache)
    else:
        logger.info("Download dataset from %s", dataset_path)
        personachat_file = cached_path(dataset_path)
        with open(personachat_file, "r", encoding="utf-8") as f:
            dataset = json.loads(f.read())

        responses = {}
        for lang, dials in dataset["test"].items():
            responses[lang] = []
            # Retrieve the original response
            for dial in dials:
                responses[lang].append(dial['response'])
            
            # Translate dataset
            google_lang_id = google_lang_id_map[lang.lower()]
            if google_lang_id != 'en':
                dataset["test"][lang] = collection_batch_translate(dials, google_lang_id, 'en', google_project_id, clean_up=True, use_cache=True)
             
        logger.info("Tokenize and encode the dataset")
        def tokenize(obj):
            if isinstance(obj, str):
                return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
            if isinstance(obj, dict):
                return dict((n, tokenize(o)) for n, o in obj.items())
            return list(tokenize(o) for o in obj)
        dataset = tokenize(dataset)
        
        for i, (lang, dials) in enumerate(dataset["test"].items()):
            # Add label to dataset
            for j, dial in enumerate(dials):
                dial['label'] = responses[lang][j]

        torch.save(dataset, dataset_cache)
        
    return dataset

def get_dataset(tokenizer, dataset_path, dataset_cache, lang=""):
    """ Get tokenized PERSONACHAT dataset."""
    if lang in ["En", "Fr", "It", "Id", "Jp", "Ko", "Zh"]:
        dataset_cache = dataset_cache + '_' + type(tokenizer).__name__ + "_" +lang
        print(dataset_cache)
    else:
        dataset_cache = dataset_cache + '_' + type(tokenizer).__name__
    if dataset_cache and os.path.isfile(dataset_cache):
        logger.info("Load tokenized dataset from cache at %s", dataset_cache)
        dataset = torch.load(dataset_cache)
    else:
        logger.info("Download dataset from %s", dataset_path)
        personachat_file = cached_path(dataset_path)
        with open(personachat_file, "r", encoding="utf-8") as f:
            dataset = json.loads(f.read())

        logger.info("Tokenize and encode the dataset")
        def tokenize(obj):
            if isinstance(obj, str):
                # print(obj)
                # print(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj)))
                return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
            if isinstance(obj, dict):
                return dict((n, tokenize(o)) for n, o in obj.items())
            return list(tokenize(o) for o in obj)
        if lang in ["En", "Fr", "It", "Id", "Jp", "Ko", "Zh"]:
            dataset_sub = {"train":{lang:dataset["train"][lang]}, "valid":{lang:dataset["valid"][lang]}, "test":{lang:dataset["test"][lang]}}
            dataset = tokenize(dataset_sub)
        else:
            dataset = tokenize(dataset)
        torch.save(dataset, dataset_cache)
    return dataset


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def make_logdir(model_name: str):
    """Create unique path to save results and checkpoints, e.g. saves/Sep22_19-45-59_gpu-7_gpt2"""
    # Code copied from ignite repo
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    logdir = os.path.join(
        'saves', current_time + '_' + socket.gethostname() + '_' + model_name)
    return logdir