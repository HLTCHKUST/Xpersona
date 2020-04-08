# Multilingual Experiments
Prepocessing:
```bash
python combine_dataset.py
```

## M-Causal Bert
Train M-Causal Bert:
```bash
python ./train_decoder_only.py --gradient_accumulation_steps=4 --max_turns=3 --train_batch_size=4 --valid_batch_size=4 --dataset_path multilingual_new.json --n_epochs 5
```
We provided the [trained M-CausalBert](https://drive.google.com/open?id=1kBRBdpf8nbfADYD0QU57zkEj2ss-WY_C) for you to skip training.

Evaluate:
```bash
python evaluate_decoder_only.py --model_checkpoint $checkpoint --max_turns=3 --no_sample --dataset_path multilingual_new.json
```

Interact:
```bash
CUDA_VISIBLE_DEVICES=1 python interact_decoder_only.py --model_checkpoint $checkpoint
```

Selfplay:
```bash
CUDA_VISIBLE_DEVICES=1 python interact_decoder_only.py --model_checkpoint $checkpoint
```

## Other Experiments
Please check **run.sh** for more details.
