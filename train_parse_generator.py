import os, argparse, h5py, codecs
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from nltk import ParentedTree
from model import SynPG
from utils import Timer, make_path, load_data, load_dictionary, deleaf, getleaf, tree2tmpl, sent2str, synt2str
from pprint import pprint

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, default="./model/", 
                       help="directory to save models")
parser.add_argument('--output_dir', type=str, default="./output_pg/",
                       help="directory to save outputs")
parser.add_argument('--dictionary_path', type=str, default="./data/dictionary.pkl", 
                       help="dictionary file")
parser.add_argument('--train_data_path', type=str, default="./data/train_data.h5",
                       help="training data")
parser.add_argument('--valid_data_path', type=str, default="./data/valid_data.h5",
                       help="validation data")
parser.add_argument('--max_sent_len', type=int, default=40,
                       help="max length of sentences")
parser.add_argument('--max_tmpl_len', type=int, default=100,
                       help="max length of templates")
parser.add_argument('--max_synt_len', type=int, default=160,
                       help="max length of syntax")
parser.add_argument('--word_dropout', type=float, default=0.2,
                       help="word dropout ratio")
parser.add_argument('--n_epoch', type=int, default=5,
                       help="number of epoches")
parser.add_argument('--batch_size', type=int, default=32,
                       help="batch size")
parser.add_argument('--lr', type=float, default=1e-4,
                       help="learning rate")
parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help="weight decay for adam")
parser.add_argument('--log_interval', type=int, default=250,
                       help="print log and validation loss evry 250 iterations")
parser.add_argument('--gen_interval', type=int, default=5000,
                       help="generate outputs every 500 iterations")
parser.add_argument('--save_interval', type=int, default=10000,
                       help="save model every 10000 iterations")
parser.add_argument('--temp', type=float, default=0.5,
                       help="temperature for generating outputs")
parser.add_argument('--seed', type=int, default=0, 
                       help="random seed")
args = parser.parse_args()
pprint(vars(args))
print()

# fix random seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.enabled = False

def train(epoch, model, train_data, valid_data, train_loader, valid_loader, optimizer, criterion, dictionary, args):
    
    timer = Timer()
    n_it = len(train_loader)
    
    for it, data_idxs in enumerate(train_loader):
        model.train()
        
        data_idxs = np.sort(data_idxs.numpy())
        
        # get batch of raw syntax
        synts_ = train_data[1][data_idxs]
        batch_size = len(synts_)
        
        # initialize tensors
        sents = np.zeros((batch_size, args.max_sent_len), dtype=np.long)    # tag sequence
        tmpls = np.zeros((batch_size, args.max_tmpl_len+2), dtype=np.long)  # template
        synts = np.zeros((batch_size, args.max_synt_len+2), dtype=np.long)  # target syntax
        
        for i in range(batch_size):
            
            # parse syntax and convert to tensor
            synt_ = synts_[i]
            synt_ = ParentedTree.fromstring(synt_)
            synt_ = deleaf(synt_)
            synt_ = [dictionary.word2idx[f"<{w}>"] for w in synt_ if f"<{w}>" in dictionary.word2idx]
            synt_ = [dictionary.word2idx["<sos>"]] + synt_ + [dictionary.word2idx["<eos>"]]
            synts[i, :len(synt_)] = synt_
            
            # parse syntax and get template
            tmpl_ = synts_[i]
            tmpl_ = ParentedTree.fromstring(tmpl_)
            tree2tmpl(tmpl_, 1, 2)
            tmpl_ = str(tmpl_).replace(")", " )").replace("(", "( ").split(" ")
            tmpl_ = [dictionary.word2idx[f"<{w}>"] for w in tmpl_ if f"<{w}>" in dictionary.word2idx]
            tmpl_ = [dictionary.word2idx["<sos>"]] + tmpl_ + [dictionary.word2idx["<eos>"]]
            tmpls[i, :len(tmpl_)] = tmpl_
            
            # get tag sequence
            sent_ = synts_[i]
            sent_ = ParentedTree.fromstring(sent_)
            sent_ = getleaf(sent_)
            sent_ = [dictionary.word2idx[f"<{w}>"] for w in sent_ if f"<{w}>" in dictionary.word2idx]
            sents[i, :len(sent_)] = sent_
            
        sents = torch.from_numpy(sents).cuda()
        tmpls = torch.from_numpy(tmpls).cuda()
        synts = torch.from_numpy(synts).cuda()
        
        # forward
        outputs = model(sents, tmpls, synts)
        
        # calculate loss
        targs_ = synts[:, 1:].contiguous().view(-1)
        outputs_ = outputs.contiguous().view(-1, outputs.size(-1))
        optimizer.zero_grad()
        loss = criterion(outputs_, targs_)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if it % args.log_interval == 0:
            # print current loss
            valid_loss = evaluate(model, valid_data, valid_loader, criterion, dictionary, args)
            print("| ep {:2d}/{} | it {:3d}/{} | {:5.2f} s | loss {:.4f} | g_norm {:.6f} | valid loss {:.4f} |".format(epoch, args.n_epoch, it, n_it, timer.get_time_from_last(), loss.item(), model.grad_norm, valid_loss))
            
        if it % args.gen_interval == 0:
            # generate output to args.output_dir
            generate(epoch, it, model, valid_data, valid_loader, dictionary, args)
            
        if it % args.save_interval == 0:
            # save model to args.model_dir
            torch.save(model.state_dict(), os.path.join(args.model_dir, "parse_generator_epoch{:02d}.pt".format(epoch)))
            
def evaluate(model, data, loader, criterion, dictionary, args):
    model.eval()
    total_loss = 0.0
    max_it = len(loader)
    with torch.no_grad():
        for it, data_idxs in enumerate(loader):
            data_idxs = np.sort(data_idxs.numpy())
        
            # get batch of raw syntax
            synts_ = data[1][data_idxs]
            batch_size = len(synts_)

            # initialize tensors
            sents = np.zeros((batch_size, args.max_sent_len), dtype=np.long)    # tag sequence
            tmpls = np.zeros((batch_size, args.max_tmpl_len+2), dtype=np.long)  # template
            synts = np.zeros((batch_size, args.max_synt_len+2), dtype=np.long)  # target syntax

            for i in range(batch_size):

                # parse syntax and convert to tensor
                synt_ = synts_[i]
                synt_ = ParentedTree.fromstring(synt_)
                synt_ = deleaf(synt_)
                synt_ = [dictionary.word2idx[f"<{w}>"] for w in synt_ if f"<{w}>" in dictionary.word2idx]
                synt_ = [dictionary.word2idx["<sos>"]] + synt_ + [dictionary.word2idx["<eos>"]]
                synts[i, :len(synt_)] = synt_

                # parse syntax and get template
                tmpl_ = synts_[i]
                tmpl_ = ParentedTree.fromstring(tmpl_)
                tree2tmpl(tmpl_, 1, 2)
                tmpl_ = str(tmpl_).replace(")", " )").replace("(", "( ").split(" ")
                tmpl_ = [dictionary.word2idx[f"<{w}>"] for w in tmpl_ if f"<{w}>" in dictionary.word2idx]
                tmpl_ = [dictionary.word2idx["<sos>"]] + tmpl_ + [dictionary.word2idx["<eos>"]]
                tmpls[i, :len(tmpl_)] = tmpl_

                # get tag sequence
                sent_ = synts_[i]
                sent_ = ParentedTree.fromstring(sent_)
                sent_ = getleaf(sent_)
                sent_ = [dictionary.word2idx[f"<{w}>"] for w in sent_ if f"<{w}>" in dictionary.word2idx]
                sents[i, :len(sent_)] = sent_

            sents = torch.from_numpy(sents).cuda()
            tmpls = torch.from_numpy(tmpls).cuda()
            synts = torch.from_numpy(synts).cuda()

            # forward
            outputs = model(sents, tmpls, synts)

            # calculate loss
            targs_ = synts[:, 1:].contiguous().view(-1)
            outputs_ = outputs.contiguous().view(-1, outputs.size(-1))
            loss = criterion(outputs_, targs_)
        
            total_loss += loss.item()
    
    return total_loss / max_it

def generate(epoch, eit, model, data, loader, dictionary, args, max_it=10):
    model.eval()
    with open(os.path.join(args.output_dir, "sents_valid_epoch{:03d}_it{:06d}.txt".format(epoch, eit)), "w") as fp:
        with torch.no_grad():
            for it, data_idxs in enumerate(loader):
                if it >= max_it:
                    break
                
                data_idxs = np.sort(data_idxs.numpy())
        
                # get batch of raw syntax
                synts_ = data[1][data_idxs]
                batch_size = len(synts_)

                # initialize tensors
                sents = np.zeros((batch_size, args.max_sent_len), dtype=np.long)    # tag sequence
                tmpls = np.zeros((batch_size, args.max_tmpl_len+2), dtype=np.long)  # template
                synts = np.zeros((batch_size, args.max_synt_len+2), dtype=np.long)  # target syntax

                for i in range(batch_size):

                    # parse syntax and convert to tensor
                    synt_ = synts_[i]
                    synt_ = ParentedTree.fromstring(synt_)
                    synt_ = deleaf(synt_)
                    synt_ = [dictionary.word2idx[f"<{w}>"] for w in synt_ if f"<{w}>" in dictionary.word2idx]
                    synt_ = [dictionary.word2idx["<sos>"]] + synt_ + [dictionary.word2idx["<eos>"]]
                    synts[i, :len(synt_)] = synt_

                    # parse syntax and get template
                    tmpl_ = synts_[i]
                    tmpl_ = ParentedTree.fromstring(tmpl_)
                    tree2tmpl(tmpl_, 1, 2)
                    tmpl_ = str(tmpl_).replace(")", " )").replace("(", "( ").split(" ")
                    tmpl_ = [dictionary.word2idx[f"<{w}>"] for w in tmpl_ if f"<{w}>" in dictionary.word2idx]
                    tmpl_ = [dictionary.word2idx["<sos>"]] + tmpl_ + [dictionary.word2idx["<eos>"]]
                    tmpls[i, :len(tmpl_)] = tmpl_

                    # get tag sequence
                    sent_ = synts_[i]
                    sent_ = ParentedTree.fromstring(sent_)
                    sent_ = getleaf(sent_)
                    sent_ = [dictionary.word2idx[f"<{w}>"] for w in sent_ if f"<{w}>" in dictionary.word2idx]
                    sents[i, :len(sent_)] = sent_

                sents = torch.from_numpy(sents).cuda()
                tmpls = torch.from_numpy(tmpls).cuda()
                synts = torch.from_numpy(synts).cuda()
                
                # generate
                idxs = model.generate(sents, tmpls, synts.size(1)-2, temp=args.temp)
                
                # write output
                for sent, tmpl, idx, synt in zip(sents.cpu().numpy(), tmpls.cpu().numpy(), idxs.cpu().numpy(), synts.cpu().numpy()):
                    fp.write(sent2str(sent, dictionary)+'\n')
                    fp.write(synt2str(tmpl[1:], dictionary)+'\n')
                    fp.write(synt2str(synt[1:], dictionary)+'\n')
                    fp.write(synt2str(idx, dictionary)+'\n')
                    fp.write("--\n")

print("==== loading data ====")

# load dictionary and data
dictionary = load_dictionary(args.dictionary_path)
train_data = load_data(args.train_data_path)
valid_data = load_data(args.valid_data_path)

train_idxs = np.arange(len(train_data[0]))
valid_idxs = np.arange(len(valid_data[0]))
print(f"number of train examples: {len(train_data[0])}")
print(f"number of valid examples: {len(valid_data[0])}")

train_loader = DataLoader(train_idxs, batch_size=args.batch_size, shuffle=True)
valid_loader = DataLoader(valid_idxs, batch_size=args.batch_size, shuffle=False)

# build model
model = SynPG(len(dictionary), 300, word_dropout=args.word_dropout)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
criterion = nn.CrossEntropyLoss(ignore_index=dictionary.word2idx["<pad>"])

model = model.cuda()
criterion = criterion.cuda()

# create folders
make_path(args.model_dir)
make_path(args.output_dir)

print("==== start training ====")
for epoch in range(1, args.n_epoch+1):
    # training
    train(epoch, model, train_data, valid_data, train_loader, valid_loader, optimizer, criterion, dictionary, args)
    # save model
    torch.save(model.state_dict(), os.path.join(args.model_dir, "parse_generator_epoch{:02d}.pt".format(epoch)))
    # shuffle training data
    train_loader = DataLoader(train_idxs, batch_size=args.batch_size, shuffle=True)
