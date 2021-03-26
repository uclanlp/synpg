import os, argparse, codecs
import numpy as np
import torch
from nltk import ParentedTree
from subwordnmt.apply_bpe import BPE, read_vocabulary
from model import SynPG
from utils import Timer, make_path, load_data, load_embedding, load_dictionary, tree2tmpl, getleaf, synt2str, reverse_bpe
from tqdm import tqdm
from pprint import pprint

parser = argparse.ArgumentParser()
parser.add_argument('--synpg_model_path', type=str, default="./model/pretrained_synpg.pt", 
                       help="prtrained SynPG")
parser.add_argument('--pg_model_path', type=str, default="./model/pretrained_parse_generator.pt", 
                       help="prtrained parse generator")
parser.add_argument('--input_path', type=str, default="./demo/input.txt",
                       help="input file")
parser.add_argument('--output_path', type=str, default="./demo/output.txt",
                       help="output file")
parser.add_argument('--bpe_codes_path', type=str, default='./data/bpe.codes',
                       help="bpe codes file")
parser.add_argument('--bpe_vocab_path', type=str, default='./data/vocab.txt',
                       help="bpe vcocabulary file")
parser.add_argument('--bpe_vocab_thresh', type=int, default=50, 
                       help="bpe threshold")
parser.add_argument('--dictionary_path', type=str, default="./data/dictionary.pkl", 
                       help="dictionary file")
parser.add_argument('--max_sent_len', type=int, default=40,
                       help="max length of sentences")
parser.add_argument('--max_tmpl_len', type=int, default=100,
                       help="max length of tempalte")
parser.add_argument('--max_synt_len', type=int, default=160,
                       help="max length of syntax")
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

templates = [
    "( ROOT ( S ( NP ) ( VP ) ( . ) ) )",
    "( ROOT ( FRAG ( SBAR ) ( . ) ) )",
    "( ROOT ( SBARQ ( WHADVP ) ( SQ ) ( . ) ) )",
    "( ROOT ( S ( SBAR ) ( , ) ( NP ) ( VP ) ( . ) ) )",
]

def template2tensor(templates, max_tmpl_len, dictionary):
    tmpls = np.zeros((len(templates), max_tmpl_len+2), dtype=np.long)
    for i, tp in enumerate(templates):
        tmpl_ = ParentedTree.fromstring(tp)
        tree2tmpl(tmpl_, 1, 2)
        tmpl_ = str(tmpl_).replace(")", " )").replace("(", "( ").split(" ")
        tmpl_ = [dictionary.word2idx[f"<{w}>"] for w in tmpl_ if f"<{w}>" in dictionary.word2idx]
        tmpl_ = [dictionary.word2idx["<sos>"]] + tmpl_ + [dictionary.word2idx["<eos>"]]
        tmpls[i, :len(tmpl_)] = tmpl_
    
    tmpls = torch.from_numpy(tmpls).cuda()
    
    return tmpls

def generate(sent, synt, tmpls, synpg_model, pg_model, args):
    with torch.no_grad():
        
        # convert syntax to tag sequence
        tagss = np.zeros((len(tmpls), args.max_sent_len), dtype=np.long)
        tags_ = ParentedTree.fromstring(synt)
        tags_ = getleaf(tags_)
        tags_ = [dictionary.word2idx[f"<{w}>"] for w in tags_ if f"<{w}>" in dictionary.word2idx]
        tagss[:, :len(tags_)] = tags_[:args.max_sent_len]
        
        tagss = torch.from_numpy(tagss).cuda()
        
        # generate parses from tag sequence and templates
        parse_idxs = pg_model.generate(tagss, tmpls, args.max_synt_len, temp=args.temp)
        
        # add <sos> and remove tokens after <eos>
        synts = np.zeros((len(tmpls), args.max_synt_len+2), dtype=np.long)
        synts[:, 0] = 1
        
        for i in range((len(tmpls))):
            parse_idx = parse_idxs[i].cpu().numpy()
            eos_pos = np.where(parse_idx==dictionary.word2idx["<eos>"])[0]
            eos_pos = eos_pos[0]+1 if len(eos_pos) > 0 else len(parse_idx)
            synts[i, 1:eos_pos+1] = parse_idx[:eos_pos]
            
        synts = torch.from_numpy(synts).cuda()
        
        # bpe segment and convert sentence to tensor
        sents = np.zeros((len(tmpls), args.max_sent_len), dtype=np.long)
        sent_ = bpe.segment(sent).split()
        sent_ = [dictionary.word2idx[w] if w in dictionary.word2idx else dictionary.word2idx["<unk>"] for w in sent_]
        sents[:, :len(sent_)] = sent_[:args.max_sent_len]
        sents = torch.from_numpy(sents).cuda()
        
        # generate paraphrases from sentence and generated parses
        output_idxs = synpg_model.generate(sents, synts, args.max_sent_len, temp=args.temp)
        output_idxs = output_idxs.cpu().numpy()
        
        paraphrases = [reverse_bpe(synt2str(output_idxs[i], dictionary).split()) for i in range(len(tmpls))]
        
        return paraphrases


print("==== loading models ====")

# load bpe codes
bpe_codes = codecs.open(args.bpe_codes_path, encoding='utf-8')
bpe_vocab = codecs.open(args.bpe_vocab_path, encoding='utf-8')
bpe_vocab = read_vocabulary(bpe_vocab, args.bpe_vocab_thresh)
bpe = BPE(bpe_codes, '@@', bpe_vocab, None)

# load dictionary and models
dictionary = load_dictionary(args.dictionary_path)

synpg_model = SynPG(len(dictionary), 300, word_dropout=0.0)
synpg_model.load_state_dict(torch.load(args.synpg_model_path))
synpg_model = synpg_model.cuda()
synpg_model.eval()

pg_model = SynPG(len(dictionary), 300, word_dropout=0.0)
pg_model.load_state_dict(torch.load(args.pg_model_path))
pg_model = pg_model.cuda()
pg_model.eval()

print("==== generate paraphrases ====")

# convert template strings to tensors
tmpls = template2tensor(templates, args.max_tmpl_len, dictionary)

with open(args.input_path) as fp:
    lines = fp.readlines()

with open(args.output_path, "w") as fp:
    for line in tqdm(lines, ascii=True):
        sent, synt = line.strip().split('\t')
        
        # generate paraphrases
        paraphrases = generate(sent, synt, tmpls, synpg_model, pg_model, args)
        
        # write to output file
        fp.write("INPUT\n")
        fp.write(sent+"\n")
        for template, paraphrase in zip(templates, paraphrases):
            fp.write("--\n")
            fp.write(template+"\n")
            fp.write(paraphrase+"\n")
        fp.write("--\n\n")
    
