## SynPG

Code for our EACL-2021 paper ["Generating Syntactically Controlled Paraphrases without Using Annotated Parallel Pairs"](https://arxiv.org/abs/2101.10579).

If you find that the code is useful in your research, please consider citing our paper.

    @inproceedings{Huang2021synpg,
        author    = {Kuan-Hao Huang and Kai-Wei Chang},
        title     = {Generating Syntactically Controlled Paraphrases without Using Annotated Parallel Pairs},
        booktitle = {Proceedings of the Conference of the European Chapter of the Association for Computational Linguistics (EACL)},
        year      = {2021},
    }

### Setup 

  - Python=3.7.10
  ```
  $ pip install -r requirements.txt
  ```
    
### Pretrained Models
  - [Pretrained SynPG](https://drive.google.com/file/d/1NAIpoVhCn_K5aSBKdW69mKoHx6vy4xW9/view?usp=sharing)
  - [Pretrained SynPG-Large](https://drive.google.com/file/d/10dy8z269KpPt60EOARUy-ba1f9tZBsaE/view?usp=sharing)
  - [Pretrained parse generator](https://drive.google.com/file/d/1V1hp7MU6GB2mseU41526dS7Zj1snAoP_/view?usp=sharing)
  
### Demo

  - Download [pretrained SynPG](https://drive.google.com/file/d/1NAIpoVhCn_K5aSBKdW69mKoHx6vy4xW9/view?usp=sharing) or [Pretrained SynPG-Large](https://drive.google.com/file/d/10dy8z269KpPt60EOARUy-ba1f9tZBsaE/view?usp=sharing) as well as [pretrained parse generator](https://drive.google.com/file/d/1V1hp7MU6GB2mseU41526dS7Zj1snAoP_/view?usp=sharing), and put them to `./model`
  - Run `scripts/demo.sh` or the following command to generate `demo/output.txt`
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

  - Download [data](https://drive.google.com/file/d/13qKcaVzRomGn4BqKhTWDDqGG-tbaaLX-/view?usp=sharing) and put them under `./data/` 
  - Download [glove.840B.300d.txt](http://nlp.stanford.edu/data/glove.840B.300d.zip) and put it under `./data/` 
  - Run `scripts/train_synpg.sh` or the following command to train SynPG
  
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
  - Run `scripts/train_parse_generator.sh` or the following command to train the parse generator
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
  
 
### Evaluating

  - Download [testing data](https://drive.google.com/file/d/1X6quKTwgr3lOJTIIi0Q8VDNW_SK3_Yjh/view?usp=sharing) and put them under `./data/` 
  - Run `scripts/eval.sh` or the following command to evaluate SynPG

  ```
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
 
  python eval_calculate_bleu.py --ref ./eval/target_sents.txt --input ./eval/outputs.txt
  ```
  
The BLEU scores should be similar to the following.

|             | MRPC |  PAN | Quora |
|-------------|:----:|:----:|:-----:|
| SynPG       | 26.2 | 27.3 |  33.2 |
| SynPG-Large | 36.2 | 27.1 |  34.7 |


### Fine-Tuning

One main advantage of SynPG is that SynPG learns the paraphrase generation model without using any paraphrase pairs. Therefore, it is possible to fine-tune SynPG with the texts (without using the ground truth paraphrases) in the target domain when those texts are available. This fine-tuning step would significantly improve the quality of paraphrase generation in the target domain, as shown in our [paper](https://arxiv.org/abs/2101.10579).

  - Download [testing data](https://drive.google.com/file/d/1X6quKTwgr3lOJTIIi0Q8VDNW_SK3_Yjh/view?usp=sharing) and put them under `./data/` 
  - Run `scripts/finetune_synpg.sh` or the following command to finetune SynPG

  ```
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
  ```
  
We can observe the significant improvement on BLEU scores from the table below.

|                 | MRPC |  PAN | Quora |
|-----------------|:----:|:----:|:-----:|
| SynPG           | 26.2 | 27.3 |  33.2 |
| SynPG-Large     | 36.2 | 27.1 |  34.7 |
| SynPG-Fine-Tune | 48.7 | 37.7 |  49.8 |
  
### Author

Kuan-Hao Huang / [@ej0cl6](https://khhuang.me/)
