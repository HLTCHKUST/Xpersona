
import os
import io
import argparse
import torch
import copy
import sys

import nltk
nltk.download('punkt')

from src.utils import bool_flag, initialize_exp, AttrDict
from src.evaluation.persona_chat import XPersona
from src.model.transformer import TransformerModel
from src.data.dictionary import Dictionary, BOS_WORD, EOS_WORD, PAD_WORD, UNK_WORD, MASK_WORD


def get_params():

    # parse parameters
    parser = argparse.ArgumentParser(description='Train on XNLG')

    # main parameters
    parser.add_argument("--exp_name", type=str, default="",
                        help="Experiment name")
    parser.add_argument("--dump_path", type=str, default="",
                        help="Experiment dump path")
    parser.add_argument("--exp_id", type=str, default="",
                        help="Experiment ID")

    parser.add_argument("--model_path", type=str, default="",
                        help="Model location")
    parser.add_argument("--saved_path", type=str, default="",
                        help="saved location")

    # data
    parser.add_argument("--data_path", type=str, default="",
                        help="Data path")
    parser.add_argument("--ds_name", type=str, default="xpersona",
                        help="name of dataset: xsumm or xgiga")
    parser.add_argument("--max_vocab", type=int, default=-1,
                        help="Maximum vocabulary size (-1 to disable)")
    parser.add_argument("--min_count", type=int, default=0,
                        help="Minimum vocabulary count")

    # batch parameters
    parser.add_argument("--max_len", type=int, default=256,
                        help="Maximum length of sentences (after BPE)")
    parser.add_argument("--max_len_q", type=int, default=256,
                        help="Maximum length of sentences (after BPE)")
    parser.add_argument("--max_len_a", type=int, default=256,
                        help="Maximum length of sentences (after BPE)")
    parser.add_argument("--max_len_e", type=int, default=256,
                        help="Maximum length of sentences (after BPE)")
    parser.add_argument("--group_by_size", type=bool_flag, default=False,
                        help="Sort sentences by size during the training")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Number of sentences per batch")
    parser.add_argument("--max_batch_size", type=int, default=0,
                        help="Maximum number of sentences per batch (used in combination with tokens_per_batch, 0 to disable)")
    parser.add_argument("--tokens_per_batch", type=int, default=-1,
                        help="Number of tokens per batch")

    # model / optimization
    parser.add_argument("--finetune_layers", type=str, default='0:_1',
                        help="Layers to finetune. 0 = embeddings, _1 = last encoder layer")
    parser.add_argument("--weighted_training", type=bool_flag, default=False,
                        help="Use a weighted loss during training")
    parser.add_argument("--dropout", type=float, default=0,
                        help="Fine-tuning dropout")
    parser.add_argument("--optimizer_e", type=str, default="adam,lr=0.0001",
                        help="Embedder (pretrained model) optimizer")
    parser.add_argument("--optimizer_p", type=str, default="adam,lr=0.0001",
                        help="Projection (classifier) optimizer")
    parser.add_argument("--optimizer", type=str, default="adam,lr=0.0001",
                        help="Projection (classifier) optimizer")                    
    parser.add_argument("--n_epochs", type=int, default=100,
                        help="Maximum number of epochs")
    parser.add_argument("--epoch_size", type=int, default=-1,
                        help="Epoch size (-1 for full pass over the dataset)")

    # debug
    parser.add_argument("--debug_train", type=bool_flag, default=False,
                        help="Use valid sets for train sets (faster loading)")
    parser.add_argument("--debug_slurm", type=bool_flag, default=False,
                        help="Debug multi-GPU / multi-node within a SLURM job")
    parser.add_argument("--sample_alpha", type=float, default=0,
                        help="Exponent for transforming word counts to probabilities (~word2vec sampling)")
    parser.add_argument("--word_pred", type=float, default=0.15,
                        help="Fraction of words for which we need to make a prediction")

    parser.add_argument("--max_dec_len", type=int, default=80,
                        help="Maximum length of target sentence (after BPE)")

    # decode with vocab

    parser.add_argument("--decode_with_vocab", type=bool_flag, default=False,
                        help="Decode with vocab")
    parser.add_argument("--decode_vocab_sizes", type=str, default="26000,20000",
                        help="decode_vocab_sizes")
    parser.add_argument("--vocab_path", type=str, default="",
                        help="vocab_path")

    # multi-gpu
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Multi-GPU - Local rank")
    parser.add_argument("--multi_gpu", type=bool_flag, default=False,
                        help="multi-gpu")

    parser.add_argument("--train_layers", type=str, default="",
                        help="train layers of encoder") 
    parser.add_argument("--n_enc_layers", type=int, default=0,
                        help="") 
    parser.add_argument("--n_dec_layers", type=int, default=0,
                        help="") 
    parser.add_argument("--fixed_embeddings", type=bool_flag, default=False,
                        help="fixed_embeddings")
    parser.add_argument("--fixed_position_embeddings", type=bool_flag, default=False,
                        help="fixed_position_embeddings")
    parser.add_argument("--fixed_lang_embeddings", type=bool_flag, default=False,
                        help="fixed_lang_embeddings")
    parser.add_argument("--fixed_task_embeddings", type=bool_flag, default=False,
                        help="fixed_task_embeddings")
    parser.add_argument("--beam_size", type=int, default=1,
                        help="")
    parser.add_argument("--no_init", type=str, default="None",
                        help="dont init with pretrained models")
    
    parser.add_argument("--train_directions", type=str, default="en-en",
                        help="")
    parser.add_argument("--eval_directions", type=str, default="",
                        help="")
    parser.add_argument("--emb_dim", type=int, default=-1,
                        help="Number of sentences per batch")
    parser.add_argument("--reload_emb", type=str, default="",
                        help="path to .vec produced by fasttext")
    parser.add_argument("--cut_dataset", type=int, default=-1,
                        help="Number of sentences in dataset. -1 for full dataset.")
    
    parser.add_argument("--device1", type=int, default=3, help="device id for the encoder")
    parser.add_argument("--device2", type=int, default=4, help="device id for the decoder")

    params = parser.parse_args()

    return params


def run_test():
    params = get_params()

    # initialize the experiment / build sentence embedder
    logger = initialize_exp(params)

    if params.tokens_per_batch > -1:
        params.group_by_size = True
    
    # check parameters
    assert os.path.isdir(params.data_path)
    assert os.path.isfile(params.saved_path)
    device = torch.device('cpu')
    reloaded = torch.load(params.saved_path, map_location=device)
    model_params = AttrDict(reloaded['params'])
    logger.info(
        "Supported languages: %s" % ", ".join(model_params.lang2id.keys()))
    params.n_langs = model_params['n_langs']
    params.id2lang = model_params['id2lang']
    params.lang2id = model_params['lang2id']

    if "enc_params" in reloaded:
        encoder_model_params = AttrDict(reloaded["enc_params"])
    elif params.n_enc_layers == model_params.n_layers or params.n_enc_layers == 0:
        encoder_model_params = model_params
    else:
        encoder_model_params = AttrDict(reloaded['params'])
        encoder_model_params.n_layers = params.n_enc_layers
        assert model_params.n_layers is not encoder_model_params.n_layers
    
    if "dec_params" in reloaded:
        decoder_model_params = AttrDict(reloaded["dec_params"])
    elif params.n_dec_layers == model_params.n_layers or params.n_dec_layers == 0:
        decoder_model_params = model_params
    else:
        decoder_model_params = AttrDict(reloaded['params'])
        decoder_model_params.n_layers = params.n_dec_layers
        assert model_params.n_layers is not decoder_model_params.n_layers
    
    params.encoder_model_params = encoder_model_params
    params.decoder_model_params = decoder_model_params

    # build dictionary / build encoder / build decoder / reload weights
    dico = Dictionary(reloaded['dico_id2word'], reloaded['dico_word2id'], reloaded['dico_counts'])

    for p in [params, encoder_model_params, decoder_model_params]:
        p.n_words = len(dico)
        p.bos_index = dico.index(BOS_WORD)
        p.eos_index = dico.index(EOS_WORD)
        p.pad_index = dico.index(PAD_WORD)
        p.unk_index = dico.index(UNK_WORD)
        p.mask_index = dico.index(MASK_WORD)

    encoder = TransformerModel(encoder_model_params, dico, is_encoder=True, with_output=False)
    decoder = TransformerModel(decoder_model_params, dico, is_encoder=False, with_output=True)

    encoder.load_state_dict(reloaded["encoder"])
    decoder.load_state_dict(reloaded["decoder"])
    
    scores = {}
    XPersona(encoder, decoder, scores, dico, params).test()

if __name__ == "__main__":
    run_test()