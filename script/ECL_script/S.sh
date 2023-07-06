export CUDA_VISIBLE_DEVICES=0


python -u run.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_96 \
  --model FiSSM \
  --data custom \
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
  --lradj type4 \
  --learning_rate 1e-4 \
  --patience 20 \
  --train_epochs 20 \
  --modes1 32 \
  --S4D_layer 1 \
  --d_model 256 \
  --ab 2 --ours


 python -u run.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_192_192 \
  --model FiSSM \
  --data custom \
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
  --patience 20 \
  --train_epochs 20 \
  --modes1 64 \
  --S4D_layer 1 \
  --d_model 256 \
  --prenorm True \
  --ab 2 --ours

  #--modes1 16



python -u run.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_336_336 \
  --model FiSSM \
  --data custom \
  --features S \
  --seq_len 720 \
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
  --patience 20 \
  --train_epochs 20 \
  --modes1 32 \
  --S4D_layer 5 \
  --d_model 128 \
  --ab 2 --ours


python -u run.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_720_720_moe0 \
  --model FiSSM \
  --data custom \
  --features S \
  --seq_len 720 \
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
  --patience 30 \
  --train_epochs 40 \
  --modes1 64 \
  --S4D_layer 4 \
  --d_model 128 \
  --ab 2 --ours

