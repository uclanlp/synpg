#!/usr/bin/env bash

python eval_generate.py \
  --test_data ./data/test_data_mrpc.h5 \
  --dictionary_path ./data/dictionary.pkl \
  --model_path ./model/pretrained_synpg.pt \
  --output_dir ./eval/ \
  --bpe_codes ./data/bpe.codes \
  --bpe_vocab ./data/vocab.txt \
  --bpe_vocab_thresh 50 \
  --max_sent_len 40 \
  --max_synt_len 160 \
  --word_dropout 0.0 \
  --batch_size 64 \
  --temp 0.5 \
  --seed 0 \
  
python eval_calculate_bleu.py