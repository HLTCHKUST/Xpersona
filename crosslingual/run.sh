
## fine-tune Xpersona on English and test on Chinese (using XNLG)
python xnlg-ft.py --exp_name xpersona --exp_id ftOnZh --dump_path ./dump --model_path ./data/pretrained_XNLG/en-zh_valid-en-zh.pth --data_path ./data/processed/XNLG --optimizer adam,lr=0.00001 --batch_size 8 --n_epochs 200 --epoch_size 3000 --max_len 120 --max_vocab 95000 --train_layers 1,10 --decode_with_vocab False --n_enc_layers 10 --n_dec_layers 6 --ds_name xpersona --train_directions en-en --eval_directions zh-zh

## fine-tune Xpersona on English and test on French (using XNLG)
python xnlg-ft.py --exp_name xpersona --exp_id ftOnFr --dump_path ./dump --model_path ./data/pretrained_XNLG/en-fr-zh_valid-en-fr.pth --data_path ./data/processed/XNLG --optimizer adam,lr=0.00001 --batch_size 8 --n_epochs 1 --epoch_size 3000 --max_len 120 --max_vocab 95000 --train_layers 1,10 --decode_with_vocab False --n_enc_layers 10 --n_dec_layers 6 --ds_name xpersona --train_directions en-en --eval_directions fr-fr

## fine-tune Xpersona on English and test on Italian (using XNLG based on XLM-R)
python xnlg-ft.py --exp_name xpersona --exp_id ftOnIt --dump_path ./dump --model_path /home/zihan/XNLG/xnlg/dump/stage2_en-it/6ns2eops7x/best-valid_en-it_mt_bleu.pth --data_path ./data/processed/XNLG --optimizer adam,lr=0.00001 --batch_size 1 --n_epochs 200 --epoch_size 3000 --max_len 120 --max_vocab 200000 --train_layers 1,5 --decode_with_vocab False --n_enc_layers 5 --n_dec_layers 3 --ds_name xpersona --train_directions en-en --eval_directions it-it 

## fine-tune Xpersona on English and test on Indonisian (using XNLG based on XLM-R)
python xnlg-ft.py --exp_name xpersona --exp_id ftOnId --dump_path ./dump --model_path /home/zihan/XNLG/xnlg/dump/stage2_en-id/debug/best-valid_en-id_mt_bleu.pth --data_path ./data/processed/XNLG --optimizer adam,lr=0.00001 --batch_size 1 --n_epochs 1 --epoch_size 3000 --max_len 120 --max_vocab 200000 --train_layers 1,5 --decode_with_vocab False --n_enc_layers 10 --n_dec_layers 6 --ds_name xpersona --train_directions en-en --eval_directions id-id --device1 6 --device2 0

## fine-tune Xpersona on English and test on Japanese (using XNLG based on XLM-R)
python xnlg-ft.py --exp_name xpersona --exp_id ftOnJp --dump_path ./dump --model_path /home/zihan/XNLG/xnlg/dump/stage2_en-jp/debug/best-valid_en-ja_mt_bleu.pth --data_path ./data/processed/XNLG --optimizer adam,lr=0.00001 --batch_size 1 --n_epochs 5 --epoch_size 500 --max_len 120 --max_vocab 200000 --train_layers 1,5 --decode_with_vocab False --n_enc_layers 10 --n_dec_layers 6 --ds_name xpersona --train_directions en-en --eval_directions jp-jp --device1 1 --device2 0

## fine-tune Xpersona on English and test on Korean (using XNLG based on XLM-R)
python xnlg-ft.py --exp_name xpersona --exp_id ftOnKo --dump_path ./dump --model_path /home/zihan/XNLG/xnlg/dump/stage2_en-ko/debug2/best-valid_en-ko_mt_bleu.pth --data_path ./data/processed/XNLG --optimizer adam,lr=0.00001 --batch_size 1 --n_epochs 4 --epoch_size 3000 --max_len 120 --max_vocab 200000 --train_layers 1,5 --decode_with_vocab False --n_enc_layers 10 --n_dec_layers 6 --ds_name xpersona --train_directions en-en --eval_directions ko-ko --device1 3 --device2 0


## test on English
python test_model.py --exp_name testonEn --dump_path ./dump --saved_path ./dump/xpersona/ftOnZh/best_zh-zh_Perplexity.pth --data_path ./data/processed/XNLG --optimizer adam,lr=0.00001 --batch_size 8 --n_epochs 200 --epoch_size 3000 --max_len 120 --max_vocab 95000 --train_layers 1,10 --decode_with_vocab False --n_enc_layers 10 --n_dec_layers 6 --ds_name xpersona --train_directions en-en --eval_directions en-en

## test on French
python test_model.py --exp_name testonFr --dump_path ./dump --saved_path ./dump/xpersona/ftOnFr/best_fr-fr_Perplexity.pth --data_path ./data/processed/XNLG --optimizer adam,lr=0.00001 --batch_size 8 --n_epochs 200 --epoch_size 3000 --max_len 120 --max_vocab 95000 --train_layers 1,10 --decode_with_vocab False --n_enc_layers 10 --n_dec_layers 6 --ds_name xpersona --train_directions en-en --eval_directions fr-fr

## test on Chinese
python test_model.py --exp_name testonZh --dump_path ./dump --saved_path ./dump/xpersona/ftOnZh/best_zh-zh_Perplexity.pth --data_path ./data/processed/XNLG --optimizer adam,lr=0.00001 --batch_size 8 --n_epochs 200 --epoch_size 3000 --max_len 120 --max_vocab 95000 --train_layers 1,10 --decode_with_vocab False --n_enc_layers 10 --n_dec_layers 6 --ds_name xpersona --train_directions en-en --eval_directions zh-zh

## test on Italian
python test_model.py --exp_name testonIt --dump_path ./dump --saved_path ./dump/xpersona/ftOnIt/best_it-it_Perplexity.1epoch.pth --data_path ./data/processed/XNLG --optimizer adam,lr=0.00001 --batch_size 8 --n_epochs 200 --epoch_size 3000 --max_len 120 --max_vocab 200000 --train_layers 1,5 --decode_with_vocab False --n_enc_layers 5 --n_dec_layers 3 --ds_name xpersona --train_directions en-en --eval_directions it-it

## test on Indonisian
python test_model.py --exp_name testonId --dump_path ./dump --saved_path ./dump/xpersona/ftOnId/best_id-id_Perplexity.pth --data_path ./data/processed/XNLG --optimizer adam,lr=0.00001 --batch_size 8 --n_epochs 200 --epoch_size 3000 --max_len 120 --max_vocab 200000 --train_layers 1,10 --decode_with_vocab False --n_enc_layers 10 --n_dec_layers 6 --ds_name xpersona --train_directions en-en --eval_directions id-id

## test on Japanese
python test_model.py --exp_name testonJa --dump_path ./dump --saved_path ./dump/xpersona/ftOnJp/best_ja-ja_Perplexity.pth --data_path ./data/processed/XNLG --optimizer adam,lr=0.00001 --batch_size 8 --n_epochs 200 --epoch_size 3000 --max_len 120 --max_vocab 200000 --train_layers 1,10 --decode_with_vocab False --n_enc_layers 10 --n_dec_layers 6 --ds_name xpersona --train_directions en-en --eval_directions jp-jp
