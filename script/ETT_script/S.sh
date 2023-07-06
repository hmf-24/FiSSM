export CUDA_VISIBLE_DEVICES=0


python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id auto_ETTh2_96_96 \
  --model FiSSM \
  --data ETTh2 \
  --features S \
  --seq_len 384 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --itr 1 \
  --lradj type1 \
  --learning_rate 1e-3 \
  --patience 15 \
  --train_epochs 50 \
  --modes1 8 \
  --d_model 128 \
  --S4D_layer 6 \
  --ab 2 \
  --ours \


python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id auto_ETTh2_96_192 \
  --model FiSSM \
  --data ETTh2 \
  --features S \
  --seq_len 192 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --itr 1 \
  --lradj type4 \
  --learning_rate 1e-3 \
  --patience 15 \
  --modes1 64 \
  --d_model 128 \
  --train_epochs 50 \
  --S4D_layer 5 \
  --ab 2 --ours \


python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id auto_ETTh2_96_336 \
  --model FiSSM \
  --data ETTh2 \
  --features S \
  --seq_len 336 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --itr 1 \
  --lradj type4 \
  --learning_rate 1e-3 \
  --patience 8 \
  --train_epochs 15 \
  --modes1 32 \
  --S4D_layer 4 \
  --ab 2 --ours \


python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id auto_ETTh2_96_720 \
  --model FiSSM \
  --data ETTh2 \
  --features S \
  --seq_len 2880 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --itr 1 \
  --lradj type4 \
  --learning_rate 1e-3 \
  --patience 1 \
  --train_epochs 1 \
  --modes1 32 \
  --S4D_layer 3 \
  --d_model 128 \
  --ab 2 --ours \

