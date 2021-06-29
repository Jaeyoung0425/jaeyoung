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

# class RNN_imdb(torch.nn.Module):
#     def __init__(self, vocabulary_len, batch_size, text_size, input_size, hidden_size, num_layers, bidirectional=False):
#         super(RNN_imdb, self).__init__()
#         self.vocabulary_len = vocabulary_len
#         self.batch_size = batch_size 
#         self.text_size = text_size 
#         self.input_size = input_size 
#         self.hidden_size = hidden_size  
        
#         self.embedding = torch.nn.Embedding(vocabulary_len+1, input_size, padding_idx= vocabulary_len) # '20992'가 패딩으로 들어감
#         self.RNN = torch.nn.RNN(input_size = input_size, # input_size = 300
#                                 hidden_size = hidden_size,
#                                 num_layers = num_layers,
#                                 bidirectional = bidirectional)
#         self.Linear1 = torch.nn.Linear(in_features = hidden_size , # hidden_size = 200 
#                                       out_features = 2) 
#         self.Linear2 = torch.nn.Linear(in_features = text_size * 2, # text_size = 400, 400*2 = 800
#                                        out_features = 2) 
        
        
#     def forward(self, context): 
#         emb_context = self.embedding(context)  
#         rnn_hidden_state = self.RNN(emb_context)[0]    
#         linear_text = self.Linear1(rnn_hidden_state) # 2*200행렬을 hidden_size인 200과 곱해서 output을 2로 만들어줌 
#         linear_text = linear_text.view([self.batch_size, self.text_size*2]) # batch_size = 100, text_size = 400
#         linear_text = self.Linear2(linear_text) # 2*800 800 
#         linear_text = F.softmax(linear_text, dim=1) 
#         return linear_text
        
        
#%%
# 일단,, optimizer에서 lr = 0.1, 0.001 로 해봤는데 안하는게 더 나음 
# num_layers; 3->10 안됨
# text_size = 200으로 줄여봤는데 안됨
# sentencepiece로 단어 분리하기
# linear 계층을 하나 더 넣어보기, 현재 linear1 = [200,2]을 [200,300]으로 바꾸고 [300,2]를 하나 더 넣기
#%% linear 하나 더 추가하기 

class RNN_imdb(torch.nn.Module):
    def __init__(self, vocabulary_len, batch_size, text_size, input_size, hidden_size, num_layers, bidirectional=False):
        super(RNN_imdb, self).__init__()
        self.vocabulary_len = vocabulary_len
        self.batch_size = batch_size 
        self.text_size = text_size 
        self.input_size = input_size 
        self.hidden_size = hidden_size  
        
        self.embedding = torch.nn.Embedding(vocabulary_len+1, input_size, padding_idx= vocabulary_len) # '20992'가 패딩으로 들어감
        self.RNN = torch.nn.RNN(input_size = input_size, # input_size = 300
                                hidden_size = hidden_size,
                                num_layers = num_layers,
                                bidirectional = bidirectional)
        self.Linear1 = torch.nn.Linear(in_features = hidden_size , # hidden_size = 200 
                                      out_features = 300) 
        self.Linear2 = torch.nn.Linear(in_features = text_size * 2, # text_size = 400, 400*2 = 800
                                       out_features = 2) 
        self.Linear3 = torch.nn.Linear(in_features = 300 , 
                                       out_features = 2) 
        
    def forward(self, context): 
        emb_context = self.embedding(context)  
        rnn_hidden_state = self.RNN(emb_context)[0]    
        linear_text = self.Linear1(rnn_hidden_state) # 2*200행렬을 hidden_size인 200과 곱해서 output을 2로 만들어줌 
        
        linear_text = self.Linear3(linear_text)
        
        linear_text = linear_text.view([self.batch_size, self.text_size*2]) # batch_size = 100, text_size = 400
        linear_text = self.Linear2(linear_text) # 2*800 800 
        linear_text = F.softmax(linear_text, dim=1) 
        return linear_text
        







