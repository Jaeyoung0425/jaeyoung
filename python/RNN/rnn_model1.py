import torch
import time
import torch.nn.functional as F
import numpy as np

#%% 
#batch_size     : size of mini batch 
#text_size      : number of words in a sentence  문장길이 = max 400 
#input_size     : size of RNN input data 
#hidden_size    : size of RNN hidden layer data  
#num_layers     : depth of RNN 

class RNN_imdb(torch.nn.Module):
    def __init__(self, vocabulary_len, batch_size, text_size, input_size, hidden_size, num_layers, bidirectional=True):
        super(RNN_imdb, self).__init__()
        self.vocabulary_len = vocabulary_len
        self.batch_size = batch_size 
        self.text_size = text_size 
        self.input_size = input_size 
        self.hidden_size = hidden_size  
        
        self.embedding = torch.nn.Embedding(vocabulary_len+1, input_size, padding_idx= vocabulary_len) # '20992'가 패딩으로 들어감
        self.LSTM = torch.nn.LSTM(input_size = input_size, # input_size = 300
                                hidden_size = hidden_size,
                                num_layers = num_layers,
                                bidirectional = bidirectional)
        self.Linear1 = torch.nn.Linear(in_features = hidden_size *2 , # hidden_size = 200 
                                      out_features = 2) 
        self.Linear2 = torch.nn.Linear(in_features = text_size * 2, # text_size = 400, 400*2 = 800
                                       out_features = 2) 
        
        
    def forward(self, context): 
        emb_context = self.embedding(context)  
        rnn_hidden_state = self.LSTM(emb_context)[0]    
        linear_text = self.Linear1(rnn_hidden_state) # 2*200행렬을 hidden_size인 200과 곱해서 output을 2로 만들어줌 
        linear_text = linear_text.view([self.batch_size, self.text_size*2]) # batch_size = 100, text_size = 400
        linear_text = self.Linear2(linear_text) # 2*800 800 
        linear_text = F.softmax(linear_text, dim=1) 
        return linear_text

#%%
# hadamard product = 행렬에서 같은 위치의 원소끼리 곱하는 것

#%%

        
        
        
        
        
