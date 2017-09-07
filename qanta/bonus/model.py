import torch
import torch.nn as nn
import torch.nn.functional as F

class BilinearSeqAttn(nn.Module):

    def __init__(self, x_size, y_size, normalize=True):
        super(BilinearSeqAttn, self).__init__()
        self.normalize = normalize
        self.linear = nn.Linear(y_size, x_size)

    def forward(self, x, y, x_mask):
        """
        Args:
            x: batch_size * length * hidden_size
            y: batch_size * length
            x_mask: batch_size * length (1 for padding)
        Output:
            alpha: batch_size * length
        """
        Wy = self.linear(y)
        xWy = x.bmm(Wy.unsqueeze(2)).squeeze(2)
        xWy.data.masked_fill_(x_mask.data, -float('inf'))
        if self.normalize:
            if self.training:
                alpha = F.log_softmax(xWy)
            else:
                alpha = F.softmax(xWy)
        else:
            alpha = xWy.exp()
        return alpha
        

class RNNReader(nn.Module):

    def __init__(self, cfg):
        super(RNNReader, self).__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.embed_size)
        self.doc_rnn = nn.LSTM(cfg.embed_size, cfg.doc_hidden_size,
                cfg.doc_num_layers, bidirectional=cfg.doc_bidirectional)
        self.query_rnn = nn.LSTM(cfg.embed_size, cfg.query_hidden_size,
                cfg.query_num_layers, bidirectional=cfg.query_bidirectional)
        doc_directions = 2 if cfg.doc_bidirectional else 1
        query_directions = 2 if cfg.query_bidirectional else 1
        self.start_attn = BilinearSeqAttn(cfg.doc_hidden_size * doc_directions,
                cfg.query_hidden_size * query_directions)
        self.end_attn = BilinearSeqAttn(cfg.doc_hidden_size * doc_directions,
                cfg.query_hidden_size * query_directions)

    def forward(self, doc, query, doc_mask, query_mask):
        """
        Args:
            doc = document word indices         [batch_size * len_d]
            query = question word indices       [batch_size * len_q]
            doc_mask = document padding mask    [batch_size * len_d]
            query_mask = question padding mask  [batch_size * len_q]
        Return:
            start_scores: scores for start positions [batch_size * len_d]
            end_scores: scores for end positions     [batch_size * len_d]
        """
        doc_embed = self.embedding(doc)
        query_embed = self.embedding(query)
        doc_hiddens = self.encode(doc_embed, self.doc_rnn)
        query_hiddens = self.encode(query_embed, self.query_rnn)
        query_hidden = query_hiddens[-1]

        start_scores = self.start_attn(doc_hiddens, query_hidden, doc_mask)
        end_scores = self.end_attn(doc_hiddens, query_hidden, doc_mask)
        return start_scores, end_scores
        
    def encode(self, xs, rnn):
        length, batch_size = xs.size()
        hidden = self.init_hidden(rnn, batch_size)
        xs_embed = self.embedding(xs)
        output, hidden = rnn(xs_embed, hidden)
        return output

    def init_hidden(self, rnn, batch_size):
        w = next(self.parameters()).data
        num_layers = rnn.num_layers
        hidden_size = rnn.hidden_size
        if rnn.bidirectional:
            hidden_size *= 2
        return (Variable(w.new(num_layers, batch_size, hidden_size).zero_()),
                Variable(w.new(num_layers, batch_size, hidden_size).zero_()))
