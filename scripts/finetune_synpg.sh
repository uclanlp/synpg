#!/usr/bin/env bash

python finetune_synpg.py \
  --model_dir ./model_finetune \
  --model_path ./model/pretrained_synpg.pt \
  --output_dir ./output_finetune \
  --bpe_codes_path ./data/bpe.codes \
  --bpe_vocab_path ./data/vocab.txt \
  --bpe_vocab_thresh 50 \
  --dictionary_path ./data/dictionary.pkl \
  --train_data_path ./data/test_data_mrpc.h5 \
  --valid_data_path ./data/test_data_mrpc.h5 \
  --max_sent_len 40 \
  --max_synt_len 160 \
  --word_dropout 0.4 \
  --n_epoch 50 \
  --batch_size 64 \
  --lr 1e-4 \
  --weight_decay 1e-5 \
  --log_interval 250 \
  --gen_interval 5000 \
  --save_interval 10000 \
  --temp 0.5 \
  --seed 0