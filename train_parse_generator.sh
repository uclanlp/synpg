#!/usr/bin/env bash

python train_parse_generator.py \
  --model_dir ./model \
  --output_dir ./output_pg \
  --dictionary_path ./data/dictionary.pkl \
  --train_data_path ./data/train_data.h5 \
  --valid_data_path ./data/valid_data.h5 \
  --max_sent_len 40 \
  --max_tmpl_len 100 \
  --max_synt_len 160 \
  --word_dropout 0.2 \
  --n_epoch 5 \
  --batch_size 32 \
  --lr 1e-4 \
  --weight_decay 1e-5 \
  --log_interval 250 \
  --gen_interval 5000 \
  --save_interval 10000 \
  --temp 0.5 \
  --seed 0