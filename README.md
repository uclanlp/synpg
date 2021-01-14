## SynPG

Code for our EACL-2021 paper "Generating Syntactically Controlled Paraphrases without Using Annotated Parallel Pairs".

If you find that the code is useful in your research, please consider citing our paper.

    @inproceedings{Huang2021synpg,
        author    = {Kuan-Hao Huang and
                     Kai-Wei Chang},
        title     = {Generating Syntactically Controlled Paraphrases without Using Annotated Parallel Pairs},
        booktitle = {Proceedings of the Conference of the European Chapter of the Association for Computational Linguistics (EACL)},
        year      = {2021},
    }

### Setup 
    $ pip install -r requirements.txt
    
### Pretrained Models

  - [Pretrained SynPG](https://drive.google.com/file/d/16jfqXUq0bojYIEv-D_-i5SunHn-Qarw5/view?usp=sharing)
  - [Pretrained parse generator](https://drive.google.com/file/d/1XkWpQC1gny6ieYCHS2HIyVXAMR0SUFqi/view?usp=sharing)
  
### Demo

  - Download [pretrained SynPG](https://drive.google.com/file/d/16jfqXUq0bojYIEv-D_-i5SunHn-Qarw5/view?usp=sharing) and [pretrained parse generator](https://drive.google.com/file/d/1XkWpQC1gny6ieYCHS2HIyVXAMR0SUFqi/view?usp=sharing), and put them to `./model`
  - Run `demo.sh` or the following command to generate `demo/output.txt`
  ```
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
  ```
    
### Training

  - Download [data](https://drive.google.com/file/d/1OrQjD-TcSR83LtTxXCVOemldwOILtn8e/view?usp=sharing) and put them under `./data/` 
  - Download [glove.840B.300d.txt](http://nlp.stanford.edu/data/glove.840B.300d.zip) and put it under `./data/` 
  - Run `train_synpg.sh` or the following command to train SynPG
  
  ```
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
  ```
  - Run `train_parse_generator.sh` or the following command to train the parse generator
  ```
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
  ```
    
### Author

Kuan-Hao Huang / [@ej0cl6](http://ej0cl6.github.io/)
