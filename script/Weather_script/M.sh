# TITAN V
export CUDA_VISIBLE_DEVICES=0


python -u run.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_96 \
  --model FiSSM \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --itr 1 \
  --ab 2 \
  --lradj type4 \
  --learning_rate 1e-3 \
  --modes1 32 \
  --patience 8 \
  --train_epochs 15 \
  --S4D_layer 4 \
  --ours


python -u run.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_192 \
  --model FiSSM \
  --data custom \
  --features M \
  --seq_len 192 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --itr 1 \
  --ab 2 \
  --lradj type4 \
  --learning_rate 1e-3 \
  --modes1 32 \
  --patience 8 \
  --train_epochs 15 \
  --S4D_layer 3 \
  --ours


python -u run.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_336 \
  --model FiSSM \
  --data custom \
  --features M \
  --seq_len 336 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --itr 1 \
  --ab 2 \
  --lradj type4 \
  --learning_rate 1e-3 \
  --patience 8 \
  --train_epochs 15 \
  --S4D_layer 4 \
  --modes1 32 \
  --ours


python -u run.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_720 \
  --model FiSSM \
  --data custom \
  --features M \
  --seq_len 720 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --itr 1 \
  --ab  2 \
  --lradj type4 \
  --learning_rate 1e-3 \
  --modes1 32 \
  --patience 8 \
  --train_epochs 15 \
  --S4D_layer 2 \
  --ours

