import torch
import time
import torch.nn.functional as F
import numpy as np

class RNN_imdb(torch.nn.Module):
    def __init__(self, vocabulary_len, input_size, hidden_size, num_layers, bidirectional=False):
        super(RNN_imdb, self).__init__()
        self.embedding = torch.nn.Embedding(vocabulary_len+1, input_size, padding_idx= vocabulary_len)
        self.RNN = torch.nn.RNN(input_size = input_size, 
                                hidden_size = hidden_size,
                                num_layers = num_layers,
                                bidirectional = bidirectional)
        self.Linear = torch.nn.Linear(in_features = hidden_size,
                                      out_features = 2)

    def forward(self, context):
        emb_context = self.embedding(context)
        rnn_hidden_state = self.RNN(emb_context)[0]
        rnn_hidden_state = rnn_hidden_state[:, -1, :]
        rnn_hidden_state = F.softmax(rnn_hidden_state)
        linear_text = self.Linear(rnn_hidden_state)
        return linear_text


# embedding, rnn, softmax, linear

