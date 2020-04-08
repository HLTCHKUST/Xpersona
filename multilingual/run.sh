#check the requirements.txt for dependency
python combine_dataset.py # run this code to generate the dataset in dailoague format.


################## Training...

# M-Causal Bert
CUDA_VISIBLE_DEVICES=5 python ./train_decoder_only.py --gradient_accumulation_steps=4 --max_turns=3 --train_batch_size=4 --valid_batch_size=4 --dataset_path multilingual_new.json --n_epochs 5
# CausalBert
CUDA_VISIBLE_DEVICES=5 python ./train_decoder_only.py --gradient_accumulation_steps=4 --max_turns=3 --train_batch_size=4 --valid_batch_size=4 --dataset_path multilingual_new.json --n_epochs 5 --train_lang Zh
CUDA_VISIBLE_DEVICES=2 python ./train_decoder_only.py --gradient_accumulation_steps=4 --max_turns=3 --train_batch_size=4 --valid_batch_size=4 --dataset_path multilingual_new.json --n_epochs 5 --train_lang Jp
CUDA_VISIBLE_DEVICES=0 python ./train_decoder_only.py --gradient_accumulation_steps=4 --max_turns=3 --train_batch_size=4 --valid_batch_size=4 --dataset_path multilingual_new.json --n_epochs 5 --train_lang It
CUDA_VISIBLE_DEVICES=2 python ./train_decoder_only.py --gradient_accumulation_steps=4 --max_turns=3 --train_batch_size=4 --valid_batch_size=4 --dataset_path multilingual_new.json --n_epochs 5 --train_lang Fr
CUDA_VISIBLE_DEVICES=2 python ./train_decoder_only.py --gradient_accumulation_steps=4 --max_turns=3 --train_batch_size=4 --valid_batch_size=4 --dataset_path multilingual_new.json --n_epochs 5 --train_lang Id
CUDA_VISIBLE_DEVICES=0 python ./train_decoder_only.py --gradient_accumulation_steps=4 --max_turns=3 --train_batch_size=4 --valid_batch_size=4 --dataset_path multilingual_new.json --n_epochs 5 --train_lang Ko
# M-Bert2Bert
CUDA_VISIBLE_DEVICES=5 python ./train_id.py --gradient_accumulation_steps=4 --max_turns=3 --train_batch_size=4 --valid_batch_size=4 --dataset_path multilingual_new.json --n_epochs 5
# Bert2Bert
CUDA_VISIBLE_DEVICES=5 python ./train_id.py --gradient_accumulation_steps=4 --max_turns=3 --train_batch_size=4 --valid_batch_size=4 --dataset_path multilingual_new.json --n_epochs 5 --train_lang Zh
CUDA_VISIBLE_DEVICES=7 python ./train_id.py --gradient_accumulation_steps=4 --max_turns=3 --train_batch_size=4 --valid_batch_size=4 --dataset_path multilingual_new.json --n_epochs 5 --train_lang Jp
CUDA_VISIBLE_DEVICES=7 python ./train_id.py --gradient_accumulation_steps=4 --max_turns=3 --train_batch_size=4 --valid_batch_size=4 --dataset_path multilingual_new.json --n_epochs 5 --train_lang It
CUDA_VISIBLE_DEVICES=7 python ./train_id.py --gradient_accumulation_steps=4 --max_turns=3 --train_batch_size=4 --valid_batch_size=4 --dataset_path multilingual_new.json --n_epochs 5 --train_lang Fr
CUDA_VISIBLE_DEVICES=7 python ./train_id.py --gradient_accumulation_steps=4 --max_turns=3 --train_batch_size=4 --valid_batch_size=4 --dataset_path multilingual_new.json --n_epochs 5 --train_lang Id
CUDA_VISIBLE_DEVICES=5 python ./train_id.py --gradient_accumulation_steps=4 --max_turns=3 --train_batch_size=4 --valid_batch_size=4 --dataset_path multilingual_new.json --n_epochs 5 --train_lang Ko

################### Test...

# M-Causal Bert
CUDA_VISIBLE_DEVICES=5 python evaluate_decoder_only.py --model_checkpoint saves/latest_multi-bert_decoder_only --max_turns=3 --no_sample --dataset_path multilingual_new.json
# Causal Bert
CUDA_VISIBLE_DEVICES=5 python evaluate_decoder_only.py --model_checkpoint saves/Jan26_11-16-48_black-rack-1_multi-bert_decoder_only_Zh --test_lang Zh --max_turns=3 --no_sample --dataset_path multilingual_new.json
CUDA_VISIBLE_DEVICES=5 python evaluate_decoder_only.py --model_checkpoint saves/Jan26_15-41-59_black-rack-1_multi-bert_decoder_only_Jp --test_lang Jp --max_turns=3 --no_sample --dataset_path multilingual_new.json
CUDA_VISIBLE_DEVICES=5 python evaluate_decoder_only.py --model_checkpoint saves/Jan26_20-25-46_black-rack-1_multi-bert_decoder_only_It --test_lang It --max_turns=3 --no_sample --dataset_path multilingual_new.json
CUDA_VISIBLE_DEVICES=1 python evaluate_decoder_only.py --model_checkpoint saves/Feb01_05-03-26_black-rack-1_multi-bert_decoder_only_Id --test_lang Id --max_turns=3 --no_sample --dataset_path multilingual_new.json
CUDA_VISIBLE_DEVICES=1 python evaluate_decoder_only.py --model_checkpoint saves/Feb01_15-06-45_black-rack-1_multi-bert_decoder_only_Ko --test_lang Ko --max_turns=3 --no_sample --dataset_path multilingual_new.json
CUDA_VISIBLE_DEVICES=1 python evaluate_decoder_only.py --model_checkpoint saves/Jan31_20-33-31_black-rack-1_multi-bert_decoder_only_Fr --test_lang Fr --max_turns=3 --no_sample --dataset_path multilingual_new.json

# M-Bert2Bert
CUDA_VISIBLE_DEVICES=5 python evaluate.py --model_checkpoint saves/Jan03_13-45-07_black-rack-1_multi-bert_lang_id --max_turns=3 # cannot use greedy, always repeat
# Bert2Bert
CUDA_VISIBLE_DEVICES=5 python evaluate.py --model_checkpoint saves/Jan07_20-04-39_black-rack-1_multi-bert_En --max_turns=3 --test_lang En --dataset_path multilingual_new.json


################### Iteract...
# M-Causal Bert
CUDA_VISIBLE_DEVICES=1 python interact_decoder_only.py --model_checkpoint saves/latest_multi-bert_decoder_only
# M-Causal Bert Self play 
CUDA_VISIBLE_DEVICES=1 python interact_decoder_only.py --model_checkpoint saves/latest_multi-bert_decoder_only --self_play
# Causal Bert 
CUDA_VISIBLE_DEVICES=2 python interact_decoder_only.py --model_checkpoint saves/Jan26_11-16-48_black-rack-1_multi-bert_decoder_only_Zh --test_lang Zh
# Causal Bert Self play 
CUDA_VISIBLE_DEVICES=2 python interact_decoder_only.py --model_checkpoint saves/Jan26_11-16-48_black-rack-1_multi-bert_decoder_only_Zh --test_lang Zh --self_play
# M-Bert2Bert
CUDA_VISIBLE_DEVICES=7 python interact_id.py --model_checkpoint saves/Jan03_13-45-07_black-rack-1_multi-bert_lang_id
# M-Bert2Bert Self play 
CUDA_VISIBLE_DEVICES=7 python interact_id.py --model_checkpoint saves/Jan03_13-45-07_black-rack-1_multi-bert_lang_id --self_play
