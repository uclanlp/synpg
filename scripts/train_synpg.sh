#!/usr/bin/env bash

python train_synpg.py \
  --model_dir ./model \
  --output_dir ./output \
  --bpe_codes_path ./data/bpe.codes \
  --bpe_vocab_path ./data/vocab.txt \
  --bpe_vocab_thresh 50 \
  --dictionary_path ./data/dictionary.pkl \
  --train_data_path ./data/train_data.h5 \
  --valid_data_path ./data/valid_data.h5 \
  --emb_path ./data/glove.840B.300d.txt \
  --max_sent_len 40 \
  --max_synt_len 160 \
  --word_dropout 0.4 \
  --n_epoch 5 \
  --batch_size 64 \
  --lr 1e-4 \
  --weight_decay 1e-5 \
  --log_interval 250 \
  --gen_interval 5000 \
  --save_interval 10000 \
  --temp 0.5 \
  --seed 0