
from logging import getLogger
import os
import copy
import time
import json
import math
import numpy as np
from collections import OrderedDict, defaultdict

import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
# import rouge
from nlgeval import NLGEval

from ..optim import get_optimizer
from ..utils import concat_batches, truncate, to_cuda, mask_out_v2
from ..data.dataset import TripleDataset, ParallelDataset
from ..data.loader import load_binarized, set_dico_parameters

XPersona_LANGS = ["en", "zh"]
# XPersona_LANGS = ["en", "fr"]
# XPersona_LANGS = ["en", "it"]
# XPersona_LANGS = ["en", "id"]
# XPersona_LANGS = ["en", "jp"]
# XPersona_LANGS = ["en", "ko"]

logger = getLogger()
# evaluator = rouge.Rouge(
#   metrics=['rouge-n', 'rouge-l'],
#   max_n=2,
#   limit_length=True,
#   length_limit=100,
#   length_limit_type='words',
#   alpha=0.5, # Default F1_score
#   weight_factor=1.2,
#   stemming=False)
nlgeval = NLGEval(
  no_skipthoughts=True,no_glove=True,metrics_to_omit=['CIDEr'])


def get_parameters(model, train_layers_str):
  ret = []

  fr, to = map(int, train_layers_str.split(","))
  assert fr >= 0
  if fr == 0:
    # add embeddings
    ret += model.embeddings.parameters()
    logger.info("Adding embedding parameters")
    ret += model.position_embeddings.parameters()
    logger.info("Adding positional embedding parameters")
    ret += model.lang_embeddings.parameters()
    logger.info("Adding language embedding parameters")
    fr = 1
  assert fr <= to
  # add attention layers
  # NOTE cross attention is not added
  for i in range(fr, to + 1):
    ret += model.attentions[i-1].parameters()
    ret += model.layer_norm1[i-1].parameters()
    ret += model.ffns[i-1].parameters()
    ret += model.layer_norm2[i-1].parameters()
    logger.info("Adding layer-%s parameters to optimizer" % i)

  return ret


def tokens2words(toks):
  words = []
  for tok in toks:
    if len(words) > 0 and words[-1].endswith("@@"):
      words[-1] = words[-1][:-2] + tok
    else:
      words.append(tok)
  return words


class XPersona(object):

  def __init__(self, encoder, decoder, scores, dico, params):
    self.encoder = encoder
    self.decoder = decoder
    self.params = params
    self.scores = scores
    self.dico = dico

    self.iter_cache = {}
  
  def setup_vocab_mask(self, dico):
    n_words = len(dico)
    params = self.params

    self.vocab_mask = {}

    decode_vocab_sizes = [int(s) for s in params.decode_vocab_sizes.split(",")]
    assert len(decode_vocab_sizes) == len(XPersona_LANGS)

    for lang, sz in zip(XPersona_LANGS, decode_vocab_sizes):
      
      fn = os.path.join(params.vocab_path, lang + ".vocab")
      assert os.path.isfile(fn), fn

      mask = torch.ByteTensor(n_words)
      mask.fill_(0)
      assert mask.sum() == 0
      mask[dico.eos_index] = 1
      # TODO generate unk?
      mask[dico.unk_index] = 1
      count = 0
      with open(fn) as fp:
        for line, _ in zip(fp, range(sz)):
          tok = line.strip().split("\t")[0].split(" ")[0]
          if tok not in dico.word2id:
            # logger.warn("Token %s not in dico" % tok)
            count += 1
          else: mask[dico.word2id[tok]] = 1
      
      # mask[dico.word2id["<@@"]] = 0
      logger.warn("%d tokens not in dico" % count)
      self.vocab_mask[lang] = mask
  
  def gen_references_v2(self, dico, eval_directions):
    self.references = {}
    for split in ["valid", "test"]:
      self.references[split] = {}
      for direction in eval_directions:
        x_lang, y_lang = direction
        if y_lang in self.references: continue
        refs = []
        for batch in self.get_iterator(split, x_lang, y_lang):
          _, (sent_y, len_y), _ = batch
          for j in range(len(len_y)):
            ref_sent = sent_y[1:len_y[j]-1,j]
            ref_toks = [dico[ref_sent[k].item()] for k in range(len(ref_sent))]
            ref_words = tokens2words(ref_toks)

            #zh or en2zh
            if y_lang.endswith("zh"): refs.append(" ".join("".join(ref_words)))
            else: refs.append(" ".join(ref_words))

        self.references[split][y_lang] = refs
  
  def _parse_lang(self, lang):
    if type(lang) == tuple:
      assert len(lang) == 2
      lang1, lang2 = lang
      assert lang1 in XPersona_LANGS
      assert lang2 in XPersona_LANGS
      return (lang1, lang2)
    if type(lang) == str:
      if lang in XPersona_LANGS:
        return (lang, lang)
      else:
        lang1, lang2 = lang.split("2")
        assert lang1 in XPersona_LANGS
        assert lang2 in XPersona_LANGS
        return (lang1, lang2)

  def get_iterator(self, splt, x_lang, y_lang):
    x_lang = self._parse_lang(x_lang)
    y_lang = self._parse_lang(y_lang)
    logger.info("Getting iterator -- x_lang: (%s, %s), y_lang: (%s, %s) split:%s" % (
      x_lang[0], x_lang[1], y_lang[0], y_lang[1], splt))
    return self.get_or_load_data(x_lang, y_lang, splt).get_iterator(
      shuffle=(splt == 'train'),
      group_by_size=self.params.group_by_size,
      return_indices=True)
  
  def next_batch(self, splt, x_lang, y_lang):
    
    key = (splt, x_lang, y_lang)
    if key not in self.iter_cache:
      self.iter_cache[key] = self.get_iterator(splt, x_lang, y_lang)
    try:
      ret = next(self.iter_cache[key])
    except StopIteration:
      self.iter_cache[key] = self.get_iterator(splt, x_lang, y_lang)
      ret = next(self.iter_cache[key])
    return ret
  
  def lang2str(self, lang):
    lang1, lang2 = lang
    if lang1 == lang2: return lang1
    return "%s-%s" % (lang1, lang2)
  
  def get_or_load_data(self, x_lang, y_lang, splt):
    params = self.params
    data = self.data

    lang = (x_lang, y_lang)
    if lang in self.data:
      if splt in self.data[lang]: return self.data[lang][splt]
    else:
      self.data[lang] = {}
    
    dpath = os.path.join(params.data_path, "eval", params.ds_name)

    x = load_binarized(os.path.join(dpath, "%s.x.%s.pth" % (
      splt, self.lang2str(x_lang))), params)
    y = load_binarized(os.path.join(dpath, "%s.y.%s.pth" % (
      splt, self.lang2str(y_lang))), params)
    data["dico"] = data.get("dico", x["dico"])
    set_dico_parameters(params, data, x["dico"])
    set_dico_parameters(params, data, y["dico"])

    data[lang][splt] = ParallelDataset(
      x["sentences"], x["positions"],
      y["sentences"], y["positions"],
      params)
    data[lang][splt].remove_empty_sentences()
    data[lang][splt].cut_long_sentences(params.max_len, params.max_len)

    if params.cut_dataset > 0 and splt == "train":
      data[lang][splt].select_data(0, params.cut_dataset + 1)

    return self.data[lang][splt]

  def run(self):
    params = self.params

    train_directions = [d.split("-") for d in params.train_directions.split(",")]
    eval_directions = [d.split("-") for d in params.eval_directions.split(",")]

    self.data = {}
  
    # self.encoder.cuda()
    # self.decoder.cuda()
    self.encoder.to(params.device1)
    self.decoder.to(params.device2)

    parameters = []
    if params.train_layers == "all":
      parameters.extend([_ for _ in self.encoder.parameters()])
      parameters.extend([_ for _ in self.decoder.parameters()])
    elif params.train_layers == "decoder":
      parameters = self.decoder.parameters()
    elif params.train_layers == "encoder":
      parameters = self.encoder.parameters()
    else:
      parameters = get_parameters(self.encoder, params.train_layers)
    self.optimizer = get_optimizer(parameters, params.optimizer)

    self.gen_references_v2(self.dico, eval_directions)
    if self.params.decode_with_vocab: self.setup_vocab_mask(self.dico)

    # self.best_scores = defaultdict(float)
    self.best_ppl = 1000000
    
    for epoch in range(params.n_epochs):
      self.epoch = epoch
      logger.info("XPersona - Training epoch %d ..." % epoch)
      self.train(train_directions)
      logger.info("XPersona - Evaluating epoch %d ..." % epoch)
      self.eval(eval_directions, "valid", True)
      self.eval(eval_directions, "test", False)
  
  def gen_resp_(self):
    params = self.params

    train_directions = [d.split("-") for d in params.train_directions.split(",")]
    eval_directions = [d.split("-") for d in params.eval_directions.split(",")]

    self.data = {}
  
    self.encoder.cuda()
    self.decoder.cuda()
    parameters = []
    if params.train_layers == "all":
      parameters.extend([_ for _ in self.encoder.parameters()])
      parameters.extend([_ for _ in self.decoder.parameters()])
    elif params.train_layers == "decoder":
      parameters = self.decoder.parameters()
    elif params.train_layers == "encoder":
      parameters = self.encoder.parameters()
    else:
      parameters = get_parameters(self.encoder, params.train_layers)
    self.optimizer = get_optimizer(parameters, params.optimizer)

    self.gen_references_v2(self.dico, eval_directions)
    if self.params.decode_with_vocab: self.setup_vocab_mask(self.dico)

    # self.best_scores = defaultdict(float)
    self.best_ppl = 0
    self.generate_response(eval_directions, "test")
  
  def test(self):
    params = self.params

    # train_directions = [d.split("-") for d in params.train_directions.split(",")]
    eval_directions = [d.split("-") for d in params.eval_directions.split(",")]

    self.data = {}
  
    # self.encoder.cuda()
    # self.decoder.cuda()
    self.encoder.to(params.device1)
    self.decoder.to(params.device2)
    parameters = []
    if params.train_layers == "all":
      parameters.extend([_ for _ in self.encoder.parameters()])
      parameters.extend([_ for _ in self.decoder.parameters()])
    elif params.train_layers == "decoder":
      parameters = self.decoder.parameters()
    elif params.train_layers == "encoder":
      parameters = self.encoder.parameters()
    else:
      parameters = get_parameters(self.encoder, params.train_layers)
    self.optimizer = get_optimizer(parameters, params.optimizer)

    self.gen_references_v2(self.dico, eval_directions)
    if self.params.decode_with_vocab: self.setup_vocab_mask(self.dico)

    # self.best_scores = defaultdict(float)
    self.best_ppl = 0
    self.epoch = 100
    self.eval(eval_directions, "test", False)
  
  def generate_response(self, eval_directions, split="test"):
    params = self.params
    encoder = self.encoder
    decoder = self.decoder
    encoder.eval()
    decoder.eval()
    dico = self.dico
    # best_scores = self.best_scores

    for direction in eval_directions:
      x_lang, y_lang = direction
      logger.info("Evaluating %s-%s-xpersona on %s set" % (x_lang, y_lang, split))

      X, Y = [], []
      x_lang_id = params.lang2id[x_lang[-2:]]
      y_lang_id = params.lang2id[y_lang[-2:]]
      vocab_mask=self.vocab_mask[y_lang[-2:]] if params.decode_with_vocab else None

      perplexity_list = []
      for batch in self.get_iterator(split, x_lang, y_lang):
        (sent_x, len_x), (sent_y, len_y), _ = batch
        lang_x = sent_x.clone().fill_(x_lang_id)
        lang_y = sent_y.clone().fill_(y_lang_id)

        sent_x, len_x, lang_x, sent_y, len_y, lang_y = to_cuda(sent_x, len_x, lang_x, sent_y, len_y, lang_y)

        with torch.no_grad():
          encoded = encoder(
            "fwd", x=sent_x, lengths=len_x, langs=lang_x, causal=False)
          encoded = encoded.transpose(0, 1)
          
          # calculate perplexity
          alen = torch.arange(len_y.max(), dtype=torch.long, device=len_y.device)
          pred_mask = alen[:, None] < len_y[None] - 1
          y = sent_y[1:].masked_select(pred_mask[:-1])

          if params.beam_size == 1:
            decoded, _ = decoder.generate(
              encoded, len_x, y_lang_id, max_len=params.max_dec_len,
              vocab_mask=vocab_mask)
          else:
            decoded, _ = decoder.generate_beam(
              encoded, len_x, y_lang_id, beam_size=params.beam_size,
              length_penalty=0.9, early_stopping=False,
              max_len=params.max_dec_len, vocab_mask=vocab_mask)
      
        for j in range(decoded.size(1)):
          sent = decoded[:, j]
          delimiters = (sent == params.eos_index).nonzero().view(-1)
          assert len(delimiters) >= 1 and delimiters[0].item() == 0
          sent = sent[1:] if len(delimiters) == 1  else sent[1: delimiters[1]]

          trg_tokens = [dico[sent[k].item()] for k in range(len(sent))]
          trg_words = tokens2words(trg_tokens)
          if y_lang.endswith("zh"): Y.append(" ".join("".join(trg_words)))
          else: Y.append(" ".join(trg_words))

          # if len(X) < 5:
          x_sent = sent_x[1:len_x[j], j]
          x_toks = [dico[x_sent[k].item()] for k in range(len(x_sent))]
          x_words = tokens2words(x_toks)
          X.append(x_words)
    
      logger.info("%d res %d ref" % (len(Y), len(self.references[split][y_lang])))
      for i in range(len(X)):
        logger.info("%d X: %s\nGenerated: %s\nReference: %s\n" % (
            i, " ".join(X[i]), Y[i], self.references[split][y_lang][i]))
      
  def train(self, train_directions):
    params = self.params
    encoder = self.encoder
    decoder = self.decoder

    encoder.train()
    decoder.train()

    # training variables
    losses = []
    ns = 0  # number of sentences
    nw = 0  # number of words
    t = time.time()

    # x_lang, y_lang = train_direction
    # x_lang_id = params.lang2id[x_lang[-2:]]
    # y_lang_id = params.lang2id[y_lang[-2:]]
    n_train_drt = len(train_directions)

    for step_idx in range(params.epoch_size):

      x_lang, y_lang = train_directions[step_idx % n_train_drt]
      x_lang_id = params.lang2id[x_lang[-2:]]
      y_lang_id = params.lang2id[y_lang[-2:]]

      batch = self.next_batch("train", x_lang, y_lang)
      (sent_x, len_x), (sent_y, len_y), _ = batch
      lang_x = sent_x.clone().fill_(x_lang_id)
      lang_y = sent_y.clone().fill_(y_lang_id)
      alen = torch.arange(len_y.max(), dtype=torch.long, device=len_y.device)
      pred_mask = alen[:, None] < len_y[None] - 1
      y = sent_y[1:].masked_select(pred_mask[:-1])
      assert len(y) == (len_y-1).sum().item()

      # sent_x, len_x, lang_x, sent_y, len_y, lang_y, y = to_cuda(
        # sent_x, len_x, lang_x, sent_y, len_y, lang_y, y)
      
      sent_x, len_x, lang_x = sent_x.to(params.device1), len_x.to(params.device1), lang_x.to(params.device1)
      sent_y, len_y, lang_y, y = sent_y.to(params.device2), len_y.to(params.device2), lang_y.to(params.device2), y.to(params.device2)

      enc_x = self.encoder("fwd", x=sent_x, lengths=len_x, langs=lang_x, causal=False)
      enc_x = enc_x.transpose(0, 1)

      enc_x, len_x = enc_x.to(params.device2), len_x.to(params.device2)

      dec_y = self.decoder('fwd', x=sent_y, lengths=len_y, langs=lang_y,
        causal=True, src_enc=enc_x, src_len=len_x)
      
      _, loss = self.decoder("predict", tensor=dec_y, pred_mask=pred_mask, y=y,
        get_scores=False)
      
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

      bs = len(len_y)
      ns += bs
      nw += len_y.sum().item()
      losses.append(loss.item())

      # log
      if ns % (100 * bs) < bs:
        logger.info(
          "XPersona - Epoch %i - Train iter %7i - %.1f words/s - Loss: %.4f" % (
            self.epoch, ns, nw / (time.time() - t), sum(losses) / len(losses)))
        nw, t = 0, time.time()
        losses = []
  
  def eval(self, eval_directions, split="test", save=True):
    params = self.params
    encoder = self.encoder
    decoder = self.decoder
    encoder.eval()
    decoder.eval()
    dico = self.dico
    # best_scores = self.best_scores

    for direction in eval_directions:
      x_lang, y_lang = direction
      logger.info("Evaluating %s-%s-xpersona on %s set" % (x_lang, y_lang, split))

      X, Y = [], []
      x_lang_id = params.lang2id[x_lang[-2:]]
      y_lang_id = params.lang2id[y_lang[-2:]]
      vocab_mask=self.vocab_mask[y_lang[-2:]] if params.decode_with_vocab else None

      # perplexity_list = []
      loss_list = []
      for batch in self.get_iterator(split, x_lang, y_lang):
        (sent_x, len_x), (sent_y, len_y), _ = batch
        lang_x = sent_x.clone().fill_(x_lang_id)
        lang_y = sent_y.clone().fill_(y_lang_id)

        # sent_x, len_x, lang_x, sent_y, len_y, lang_y = to_cuda(sent_x, len_x, lang_x, sent_y, len_y, lang_y)
        
        sent_x, len_x, lang_x = sent_x.to(params.device1), len_x.to(params.device1), lang_x.to(params.device1)
        sent_y, len_y, lang_y = sent_y.to(params.device2), len_y.to(params.device2), lang_y.to(params.device2)

        with torch.no_grad():
          encoded = encoder(
            "fwd", x=sent_x, lengths=len_x, langs=lang_x, causal=False)
          encoded = encoded.transpose(0, 1)
          
          encoded, len_x = encoded.to(params.device2), len_x.to(params.device2)

          # calculate perplexity
          alen = torch.arange(len_y.max(), dtype=torch.long, device=len_y.device)
          pred_mask = alen[:, None] < len_y[None] - 1
          y = sent_y[1:].masked_select(pred_mask[:-1])

          dec_y = self.decoder('fwd', x=sent_y, lengths=len_y, langs=lang_y, causal=True, src_enc=encoded, src_len=len_x)
          _, loss = self.decoder("predict", tensor=dec_y, pred_mask=pred_mask, y=y, get_scores=False)

          loss_list.append(loss.item())
          # perplexity = math.exp(loss.item())
          # perplexity_list.append(perplexity)

          if params.beam_size == 1:
            decoded, _ = decoder.generate(
              encoded, len_x, y_lang_id, max_len=params.max_dec_len,
              vocab_mask=vocab_mask)
          else:
            decoded, _ = decoder.generate_beam(
              encoded, len_x, y_lang_id, beam_size=params.beam_size,
              length_penalty=0.9, early_stopping=False,
              max_len=params.max_dec_len, vocab_mask=vocab_mask)
      
        for j in range(decoded.size(1)):
          sent = decoded[:, j]
          delimiters = (sent == params.eos_index).nonzero().view(-1)
          assert len(delimiters) >= 1 and delimiters[0].item() == 0
          sent = sent[1:] if len(delimiters) == 1  else sent[1: delimiters[1]]

          trg_tokens = [dico[sent[k].item()] for k in range(len(sent))]
          trg_words = tokens2words(trg_tokens)
          if y_lang.endswith("zh"): Y.append(" ".join("".join(trg_words)))
          else: Y.append(" ".join(trg_words))

          # if len(X) < 5:
          x_sent = sent_x[1:len_x[j], j]
          x_toks = [dico[x_sent[k].item()] for k in range(len(x_sent))]
          x_words = tokens2words(x_toks)
          X.append(x_words)
    
      logger.info("%d res %d ref" % (len(Y), len(self.references[split][y_lang])))
      for i in range(5):
        logger.info("%d X: %s\nGenerated: %s\nReference: %s\n" % (
            i, " ".join(X[i]), Y[i], self.references[split][y_lang][i]))
      eval_res = nlgeval.compute_metrics([self.references[split][y_lang][:len(Y)]], Y)
      # eval_res = evaluator.get_scores(Y, self.references[y_lang][:len(Y)])

      direction_str = "-".join(direction)

      # if save:
      #   if eval_res["Bleu_4"] > best_scores[direction_str]:
      #     logger.info("New best Bleu_4 score: %.5f! Saving model..." % eval_res["Bleu_4"])
      #     best_scores[direction_str] = eval_res["Bleu_4"]
      #     self.save("best_%s_Bleu_4" % direction_str)
      
      # use perplexity to stop train
      # calculate perplexity
      avg_loss = np.mean(loss_list)
      perplexity = math.exp(avg_loss)
      if save:
        if perplexity < self.best_ppl:
          logger.info("New best Perplexity: %.5f! Saving model..." % perplexity)
          self.best_ppl = perplexity
          self.save("best_%s_Perplexity" % direction_str)
      
      if split == "test":
        print("writing down output and refenerences ....")
        assert len(X) == len(self.references[split][y_lang][:len(Y)]) == len(Y)

        with open(os.path.join(self.params.dump_path, "output_"+str(self.epoch)+"_"+str(y_lang)+".txt"), "w") as f:
          for sent in Y:
            f.write(sent + "\n")
        with open(os.path.join(self.params.dump_path, "ref_"+str(self.epoch)+"_"+str(y_lang)+".txt"), "w") as f:
          for sent in self.references[split][y_lang][:len(Y)]:
            f.write(sent + "\n")
        with open(os.path.join(self.params.dump_path, "persona_chat_"+str(self.epoch)+"_"+str(y_lang)+".txt"), "w") as f:
          for persona_and_history, output, reference in zip(X, Y, self.references[split][y_lang][:len(Y)]):
            f.write("=====================================================\n")
            f.write("History:\n")
            f.write(" ".join(persona_and_history))
            f.write('\n')
            f.write("Response: ")
            f.write(output)
            f.write('\n')
            f.write("Ref: ")
            f.write(reference)
            f.write('\n')
      # logger.info("XPersona - %s - Epoch %d - Best BLEU-4: %.5f - scores: %s" % (
      #   direction_str, self.epoch, best_scores[direction_str], eval_res))
      
      logger.info("XPersona - %s - Epoch %d - Current Perplexity %.5f - Best Perplexity: %.5f - Metrics Scores: %s" % (
        direction_str, self.epoch, perplexity, self.best_ppl, eval_res))
      
      # eval_res_print = {metric:results["f"] for metric, results in sorted(eval_res.items(), key=lambda x: x[0])}

      # logger.info("XPersona - %s - Epoch %d - Best rouge-l: %.5f - scores: %s" % (
      #   direction_str, self.epoch, best_scores[direction_str], eval_res_print))
      
      # if eval_res["rouge-l"]['f'] > best_scores[direction_str]:
      #   logger.info("New best rouge-l score! Saving model...")
      #   best_scores[direction_str] = eval_res["rouge-l"]['f']
      #   self.save("best_%s_rouge-l" % direction_str)

  def save(self, name):
    path = os.path.join(self.params.dump_path, "%s.pth" % name)
    logger.info("Saving %s to %s ..." % (name, path))
    data = {
      "epoch": getattr(self, "epoch", 0),
      "encoder": self.encoder.state_dict(),
      "decoder": self.decoder.state_dict(),
      "enc_params": {
        k: v for k, v in self.params.encoder_model_params.__dict__.items()},
      "dec_params": {
        k: v for k, v in self.params.decoder_model_params.__dict__.items()},
      "dico_id2word": self.dico.id2word,
      "dico_word2id": self.dico.word2id,
      "dico_counts": self.dico.counts,
      "params": {k: v for k, v in self.params.__dict__.items()}
    }
    torch.save(data, path)