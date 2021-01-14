import os, errno
import h5py, pickle
import numpy as np
from datetime import datetime
from collections import defaultdict
from nltk import ParentedTree

def make_path(path):
	try:
		os.makedirs(path)
	except OSError as exc:
		if exc.errno == errno.EEXIST and os.path.isdir(path):
			pass
		else: raise

class Timer:
    def __init__(self):
        self.start_time = datetime.now()
        self.last_time = self.start_time

    def get_time_from_last(self, update=True):
        now_time = datetime.now()
        diff_time = now_time - self.last_time
        if update:
            self.last_time = now_time
        return diff_time.total_seconds()
    
    def get_time_from_start(self, update=True):
        now_time = datetime.now()
        diff_time = now_time - self.start_time
        if update:
            self.last_time = now_time
        return diff_time.total_seconds()
    
class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.word2idx["<pad>"] = 0
        self.word2idx["<sos>"] = 1
        self.word2idx["<eos>"] = 2
        self.word2idx["<unk>"] = 3
        self.word2idx["<msk>"] = 4
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.word_count = defaultdict(int)

    def add_word(self, word):
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
    
    def add_word_from_count(self, max_vocab_size):
        wc = [(self.word_count[w], w) for w in self.word_count]
        wc.sort(reverse=True)
        for c, w in wc[:max_vocab_size]:
            self.add_word(w)
        
    def __len__(self):
        return len(self.word2idx)

def load_embedding(em_path, dictionary):
    embedding = np.zeros((len(dictionary), 300))
    n_hit = 0
    with open(em_path, encoding="utf-8") as fp:
        for line in fp:
            word, values = line.strip().split(' ', 1)
            if word in dictionary.word2idx:
                embedding[dictionary.word2idx[word]] = np.array([float(v) for v in values.split(' ')])
                n_hit += 1
    print("load {} of {} from pretrained word embeddings".format(n_hit, len(dictionary)))
    print()
    return embedding

def load_dictionary(name):
    with open(name, "rb") as fp:
        dictionary = pickle.load(fp)
    return dictionary

def load_data(name):
    h5f = h5py.File(name, "r")
    data = (h5f["sents"], h5f["synts"])
    return data

def is_paren(tok):
    return tok == ")" or tok == "("

def deleaf(tree):
    nonleaves = ''
    for w in str(tree).replace('\n', '').split():
        w = w.replace('(', '( ').replace(')', ' )')
        nonleaves += w + ' '

    arr = nonleaves.split()
    for n, i in enumerate(arr):
        if n + 1 < len(arr):
            tok1 = arr[n]
            tok2 = arr[n + 1]
            if not is_paren(tok1) and not is_paren(tok2):
                arr[n + 1] = ""

    nonleaves = " ".join(arr)
    return nonleaves.split()

def getleaf(tree):
    nonleaves = ''
    for w in str(tree).replace('\n', '').split():
        w = w.replace('(', '( ').replace(')', ' )')
        nonleaves += w + ' '
    
    leaves = []
    arr = nonleaves.split()
    for n, i in enumerate(arr):
        if n + 1 < len(arr):
            tok1 = arr[n]
            tok2 = arr[n + 1]
            if not is_paren(tok1) and not is_paren(tok2):
                leaves.append(arr[n])

    return leaves

def tree2tmpl(tree, level, mlevel):
    if level == mlevel:
        for idx, n in enumerate(tree):
            if isinstance(n, ParentedTree):
                tree[idx] = "(" + n.label() + ")"
    else:
        for n in tree:
            tree2tmpl(n, level + 1, mlevel)

def reverse_bpe(sent):
    x = []
    cache = ''

    for w in sent:
        if w.endswith('@@'):
            cache += w.replace('@@', '')
        elif cache != '':
            x.append(cache + w)
            cache = ''
        else:
            x.append(w)

    return ' '.join(x)

def sent2str(sent, dictionary):
    return " ".join([dictionary.idx2word[i] for i in sent if i != dictionary.word2idx["<pad>"]])

def synt2str(synt, dictionary):
    eos_pos = np.where(synt==dictionary.word2idx["<eos>"])[0]
    eos_pos = eos_pos[0] if len(eos_pos) > 0 else len(synt)
    return " ".join([dictionary.idx2word[i][1:-1] if i in dictionary.span_idxs else dictionary.idx2word[i] for i in synt[:eos_pos]])
