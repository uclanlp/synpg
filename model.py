import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import Transformer
    
class SynPG(nn.Module):
    def __init__(self, vocab_size, em_size, word_dropout=0.4, dropout=0.1):
        super(SynPG, self).__init__()
        self.vocab_size = vocab_size
        self.em_size = em_size
        self.word_dropout = word_dropout
        self.dropout = dropout

        # vcocabulary embedding
        self.embedding_encoder = nn.Embedding(vocab_size, em_size)
        self.embedding_decoder = nn.Embedding(vocab_size, em_size)
        
        # positional encoding
        self.pos_encoder = PositionalEncoding(em_size, dropout=0.0)

        self.transformer = Transformer(d_model=em_size, nhead=6, dropout=dropout)
        
        # linear Transformation
        self.linear = nn.Linear(em_size, vocab_size)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        
        # initialize cocabulary matrix weight
        self.embedding_encoder.weight.data.uniform_(-initrange, initrange)
        self.embedding_decoder.weight.data.uniform_(-initrange, initrange)
        
        # initialize linear weight
        self.linear.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.fill_(0)
    
    def load_embedding(self, embedding):
        self.embedding_encoder.weight.data.copy_(torch.from_numpy(embedding))
        self.embedding_decoder.weight.data.copy_(torch.from_numpy(embedding))
        
    def store_grad_norm(self, grad):
        norm = torch.norm(grad, 2, 1)
        self.grad_norm = norm.detach().data.mean()
        return grad
    
    def generate_square_mask(self, max_sent_len, max_synt_len):
        size = max_sent_len + max_synt_len + 2
        mask = torch.zeros((size, size))
        mask[:max_sent_len, max_sent_len:] = float("-inf")
        mask[max_sent_len:, :max_sent_len] = float("-inf")
        return mask
    
    def forward(self, sents, synts, targs):
        batch_size = sents.size(0)
        max_sent_len = sents.size(1)
        max_synt_len = synts.size(1) - 2  # count without <sos> and <eos>
        max_targ_len = targs.size(1) - 2  # count without <sos> and <eos>
        
        # apply word dropout
        drop_mask = torch.bernoulli(self.word_dropout*torch.ones(max_sent_len)).bool().cuda()
        sents = sents.masked_fill(drop_mask, 0)
        
        # sentence, syntax => embedding
        sent_embeddings = self.embedding_encoder(sents).transpose(0, 1) * np.sqrt(self.em_size)
        synt_embeddings = self.embedding_encoder(synts).transpose(0, 1) * np.sqrt(self.em_size)
        synt_embeddings = self.pos_encoder(synt_embeddings)
        en_embeddings = torch.cat((sent_embeddings, synt_embeddings), dim=0)
        
        # record gradient
        if en_embeddings.requires_grad:
            en_embeddings.register_hook(self.store_grad_norm)
        
        # do not allow cross attetion
        src_mask = self.generate_square_mask(max_sent_len, max_synt_len).cuda()
        
        # target => embedding
        de_embeddings = self.embedding_decoder(targs[:, :-1]).transpose(0, 1) * np.sqrt(self.em_size)
        de_embeddings = self.pos_encoder(de_embeddings)
        
        # sequential mask
        tgt_mask = self.transformer.generate_square_subsequent_mask(max_targ_len+1).cuda()
        
        # forward
        outputs = self.transformer(en_embeddings, de_embeddings, src_mask=src_mask, tgt_mask=tgt_mask)
        
        # apply linear layer to vcocabulary size
        outputs = outputs.transpose(0, 1)
        outputs = self.linear(outputs.contiguous().view(-1, self.em_size))
        outputs = outputs.view(batch_size, max_targ_len+1, self.vocab_size)
        
        return outputs
    
    def generate(self, sents, synts, max_len, sample=True, temp=0.5):
        batch_size = sents.size(0)
        max_sent_len = sents.size(1)
        max_synt_len = synts.size(1) - 2  # count without <sos> and <eos>
        max_targ_len = max_len
        
        # output index starts with <sos>
        idxs = torch.zeros((batch_size, max_targ_len+2), dtype=torch.long).cuda()
        idxs[:, 0] = 1
        
        # sentence, syntax => embedding
        sent_embeddings = self.embedding_encoder(sents).transpose(0, 1) * np.sqrt(self.em_size)
        synt_embeddings = self.embedding_encoder(synts).transpose(0, 1) * np.sqrt(self.em_size)
        synt_embeddings = self.pos_encoder(synt_embeddings)
        en_embeddings = torch.cat((sent_embeddings, synt_embeddings), dim=0)
        
        # do not allow cross attetion
        src_mask = self.generate_square_mask(max_sent_len, max_synt_len).cuda()
        
        # starting index => embedding
        de_embeddings = self.embedding_decoder(idxs[:, :1]).transpose(0, 1) * np.sqrt(self.em_size)
        de_embeddings = self.pos_encoder(de_embeddings)
        
        # sequential mask
        tgt_mask = self.transformer.generate_square_subsequent_mask(de_embeddings.size(0)).cuda()
        
        # encode
        memory = self.transformer.encoder(en_embeddings, mask=src_mask)
        
        # auto-regressively generate output
        for i in range(1, max_targ_len+2):
            # decode
            outputs = self.transformer.decoder(de_embeddings, memory, tgt_mask=tgt_mask)
            outputs = self.linear(outputs[-1].contiguous().view(-1, self.em_size))
            
            # get argmax index or sample index
            if not sample:
                values, idx = torch.max(outputs, 1)
            else:
                probs = F.softmax(outputs/temp, dim=1)
                idx = torch.multinomial(probs, 1).squeeze(1)
            
            # save to output index
            idxs[:, i] = idx
            
            # concatenate index to decoding
            de_embeddings = self.embedding_decoder(idxs[:, :i+1]).transpose(0, 1) * np.sqrt(self.em_size)
            de_embeddings = self.pos_encoder(de_embeddings)
            
            # new sequential mask
            tgt_mask = self.transformer.generate_square_subsequent_mask(de_embeddings.size(0)).cuda()
        
        return idxs[:, 1:]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
