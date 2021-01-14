#!/usr/bin/env bash

python generate.py \
  --synpg_model_path ./model/pretrained_synpg.pt \
  --pg_model_path ./model/pretrained_parse_generator.pt \
  --input_path ./demo/input.txt \
  --output_path ./demo/output.txt \
  --bpe_codes_path ./data/bpe.codes \
  --bpe_vocab_path ./data/vocab.txt \
  --bpe_vocab_thresh 50 \
  --dictionary_path ./data/dictionary.pkl \
  --max_sent_len 40 \
  --max_tmpl_len 100 \
  --max_synt_len 160 \
  --temp 0.5 \
  --seed 0
