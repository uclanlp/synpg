import os, argparse
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from pprint import pprint

parser = argparse.ArgumentParser()
parser.add_argument('--ref', type=str, default="./eval/target_sents.txt")
parser.add_argument('--input', type=str, default="./eval/outputs.txt")
args = parser.parse_args()
pprint(vars(args))
print()

def cal_bleu(hyp, ref, n):
    hyp = hyp.strip().split(' ')
    ref = ref.strip().split(' ')
    
    if n == 0:
        return sentence_bleu([ref], hyp)
    elif n == 1:
        weights = (1, 0, 0, 0)
    elif n == 2:
        weights = (0, 1, 0, 0)
    elif n == 3:
        weights = (0, 0, 1, 0)
    elif n == 4:
        weights = (0, 0, 0, 1)

    return sentence_bleu([ref], hyp, weights=weights)

with open(args.ref) as fp:
    targs = fp.readlines()

with open(args.input) as fp:
    preds = fp.readlines()
    
assert len(targs) == len(preds)

print(f"number of examples: {len(preds)}")

scores = [cal_bleu(pred, targ, 0) for pred, targ in zip(preds, targs)]

print(f"BLEU: {np.mean(scores)*100.0}")
            
