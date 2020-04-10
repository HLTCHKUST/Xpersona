# Cross-lingual Experiments
## Prepocessing:
```bash
python data_preprocessing.py
```

## XNLG Pre-training
- Two stages XNLG pre-training (Stage #1: Encoding Pre-training; Stage #2: Decoding Pre-training; More details in https://github.com/CZWin32768/XNLG).
- We directly use the provided [EN-ZH and EN-FR XNLG models](https://github.com/CZWin32768/XNLG) for Chinese and French zero-shot adaptation.
- We leverage [XLM-R Base](https://github.com/facebookresearch/XLM) as the first stage pre-trained model for other languages zero-shot adaptation.

We provided the [Pre-trained XNLG models](https://drive.google.com/open?id=1kBRBdpf8nbfADYD0QU57zkEj2ss-WY_C) for you to skip the XNLG pre-training process.

## XNLG Fine-tuning
Fine-tune pre-trained XNLG to persona chat in English and evaluate in target languages (e.g., Chinese):
```bash
python xnlg-ft.py --exp_name xpersona --exp_id ftOnZh --dump_path ./dump --model_path ./data/pretrained_XNLG/en-zh_valid-en-zh.pth --data_path ./data/processed/XNLG --optimizer adam,lr=0.00001 --batch_size 8 --n_epochs 200 --epoch_size 3000 --max_len 120 --max_vocab 95000 --train_layers 1,10 --decode_with_vocab False --n_enc_layers 10 --n_dec_layers 6 --ds_name xpersona --train_directions en-en --eval_directions zh-zh
```

Test Perplexity on the fine-tuned model (e.g., Chinese)
```bash
python test_model.py --exp_name testonZh --dump_path ./dump --saved_path ./dump/xpersona/ftOnZh/best_zh-zh_Perplexity.pth --data_path ./data/processed/XNLG --optimizer adam,lr=0.00001 --batch_size 8 --n_epochs 200 --epoch_size 3000 --max_len 120 --max_vocab 95000 --train_layers 1,10 --decode_with_vocab False --n_enc_layers 10 --n_dec_layers 6 --ds_name xpersona --train_directions en-en --eval_directions zh-zh
```

Evaluate BLEU:
```bash
bash eval_bleu.sh
```

## Other Experiments
Please check **run.sh** for more details.