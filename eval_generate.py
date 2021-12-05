import os, argparse
import h5py, codecs
import numpy as np
import torch
from torch.utils.data import DataLoader
from nltk import ParentedTree
from subwordnmt.apply_bpe import BPE, read_vocabulary
from model import SynPG
from utils import Timer, make_path, load_dictionary, deleaf, synt2str, reverse_bpe
from pprint import pprint
from tqdm import tqdm
import ipdb

parser = argparse.ArgumentParser()
parser.add_argument('--test_data', type=str, default="./data/test_data_mrpc.h5")
parser.add_argument('--dictionary_path', type=str, default="./data/dictionary.pkl")
parser.add_argument('--model_path', type=str, default="./model/pretrained_synpg.pt")
parser.add_argument('--output_dir', type=str, default="./eval/")
parser.add_argument('--bpe_codes', type=str, default='./data/bpe.codes')
parser.add_argument('--bpe_vocab', type=str, default='./data/vocab.txt')
parser.add_argument('--bpe_vocab_thresh', type=int, default=50)
parser.add_argument('--max_sent_len', type=int, default=40)
parser.add_argument('--max_synt_len', type=int, default=160)
parser.add_argument('--word_dropout', type=float, default=0.0)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--sample', action="store_true", default=False)
parser.add_argument('--temp', type=float, default=0.5)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()
pprint(vars(args))
print()

# fix random seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.enabled = False

def load_data(name):
    h5f = h5py.File(name, "r")
    data = (h5f["sents1"], 
            h5f["sents2"], 
            h5f["synts1"], 
            h5f["synts2"])

    return data

def generate(model, data, loader, dictionary, bpe, args):
    model.eval()
    with open(os.path.join(args.output_dir, f"target_sents.txt"), "w") as fp1, \
         open(os.path.join(args.output_dir, f"target_synts.txt"), "w") as fp2, \
         open(os.path.join(args.output_dir, f"outputs.txt"), "w") as fp3:
        with torch.no_grad():
            iterator = tqdm(loader, total=len(loader))
            for it, data_idxs in enumerate(iterator):
                data_idxs = data_idxs.numpy()
                
                sents_ = data[0][data_idxs] # sents1
                targs_ = data[1][data_idxs] # sents2
                synts_ = data[3][data_idxs] # synts2

                batch_size = len(sents_)
                sents = np.zeros((batch_size, args.max_sent_len), dtype=np.long)
                synts = np.zeros((batch_size, args.max_synt_len+2), dtype=np.long)

                for i in range(batch_size):
                    sent_ = sents_[i]
                    sent_ = bpe.segment(sent_).split()
                    sent_ = [dictionary.word2idx[w] if w in dictionary.word2idx else dictionary.word2idx["<unk>"] for w in sent_]
                    sents[i, :len(sent_)] = sent_

                    synt_ = synts_[i]
                    synt_ = ParentedTree.fromstring(synt_)
                    synt_ = deleaf(synt_)
                    synt_ = [dictionary.word2idx[f"<{w}>"] for w in synt_ if f"<{w}>" in dictionary.word2idx]
                    synt_ = [dictionary.word2idx["<sos>"]] + synt_ + [dictionary.word2idx["<eos>"]]
                    synts[i, :len(synt_)] = synt_

                sents = torch.from_numpy(sents).cuda()
                synts = torch.from_numpy(synts).cuda()
                
                idxs = model.generate(sents, synts, sents.size(1), sample=args.sample, temp=args.temp)
                
                for sent, idx, targ, synt_ in zip(sents_, idxs.cpu().numpy(), targs_, synts_):
                    fp1.write(targ+'\n')
                    fp2.write(synt_+'\n')
                    fp3.write(reverse_bpe(synt2str(idx, dictionary).split())+'\n')

print("==== loading data ====")
bpe_codes = codecs.open(args.bpe_codes, encoding='utf-8')
bpe_vocab = codecs.open(args.bpe_vocab, encoding='utf-8')
bpe_vocab = read_vocabulary(bpe_vocab, args.bpe_vocab_thresh)
bpe = BPE(bpe_codes, '@@', bpe_vocab, None)

dictionary = load_dictionary(args.dictionary_path)
test_data = load_data(args.test_data)
test_idxs = np.arange(len(test_data[0]))
print(f"number of valid examples: {len(test_data[0])}")
test_loader = DataLoader(test_idxs, batch_size=args.batch_size, shuffle=False)

# load model
model = SynPG(len(dictionary), 300, word_dropout=args.word_dropout)
model.load_state_dict(torch.load(args.model_path))
model = model.cuda()
model.eval()

make_path(args.output_dir)

print("==== start testing ====")
generate(model, test_data, test_loader, dictionary, bpe, args)
