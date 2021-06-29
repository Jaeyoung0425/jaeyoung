# -*- coding: utf-8 -*-
"""
Created on Tue May 11 14:22:47 2021

@author: JalynneHEO
"""
#%%
import sentencepiece as spm
import sentencepiece
import pandas as pd
import csv
import torch
#%%
with open('./result.txt', 'w', encoding='utf8') as f:
    f.write('\n')

sentencepiece.SentencePieceTrainer.Train('--input=ted_en-20160408.txt'
                               '--model_prefix= ted '
                               '--vocab_size=500000 '
                               '--model_type=bpe '
                               '--max_sentence_length=500000')
#%%
class SGNS(torch.nn.Module):
    def __init__(self, D_dim, W_dim): 
        super(SGNS, self).__init__()
        self.dict_len = D_dim
        self.linear1 = torch.nn.Linear(D_dim, W_dim)
        self.linear2 = torch.nn.Linear(D_dim, W_dim)
        
    def forward(self, number_1, number_2): #number_1 = 문장에서 중심단어의 위치
        try:
            temp_tensor = torch.Tensor([0]*self.dict_len)
            temp_tensor[number_1] = 1
            
            projection = self.linear1(temp_tensor)
            
            temp_tensor2 = torch.Tensor([0]*self.dict_len)
            temp_tensor2[number_2] = 1 
            
            projection2 = self.linear2(temp_tensor2)
            
            output = torch.dot(projection, projection2)
            output = torch.sigmoid(output)
            output = output.unsqueeze(0)
            output = torch.cat((torch.Tensor([1 - output[0].item()]).unsqueeze(0), output), dim=-1)
        
        except:
            return None
#%%
sp = spm.SentencePieceProcessor()
vocab_file = './ted.model'
#load the vocab file
sp.Load(vocab_file) #Load
#%%
with open('ted_eng.txt') as data:
   lines = data.readlines()

#%% 
result = [word_tokenize(sentence) for sentence in normalized_text]
#print(result)
#%%
result = [sp.EncodeAsIds(i) for i in lines]
#%%
model = SGNS(sp.GetPieceSize(), 50)
criterion = torch.nn.CrossEntropyLoss

optimizer = torch.optim.Adam(model.parameters(), lr=0.01) 
for sentence in tqdm(result): 
    for word_num in range(len(sentence)): 
        # for words in window size 2, label 1 and learn
        for _ in [-2,-1,1,2]:
            if word_num + _ < 0: 
                pass 
            elif word_num + _ >= len(sentence): 
                pass
            else:
                #input two number
                output = model.forward(sentence[word_num],sentence[word_num +_]) # number_1, number_2
                
                
                if output is None:
                    pass
                else:
                    output = output.unsqueeze(0)
                    loss = criterion(output, torch.LongTensor([1])) 
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        # get random 5 words for each center word
        random_words = torch.randint(0, sp.GetPieceSize(), (5,)) 
        for _ in range(5): 
            # if random word is in neighbor, pass 
            # print(random_words[_].item())
            if random_words[_].item() in sentence[max(0,word_num -2):word_num + 2]: #max를 써준 이유는 음수가 들어가면 안돼서
                pass
            #랜덤 words가 주변단어일 경우 pass
            else: 
                # output = model.forward(sentence[word_num],sentence[word_num + _]) 내가 잘못 적은 답
                output = model.forward(sentence[word_num],random_words[_].item())
                #위에서 output 이랑 여기서 output 이랑 왜 다른건 네거티브 샘플링이니까 주변단어랑 관계없는 단어들까지 전부다 학습
                
                if output is None: 
                    pass
                else:
                    output = output.unsqueeze(0)
                    loss = criterion(output, torch.LongTensor([0])) # 관련없는 단어는 라벨을 0으로 설정
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

torch.save(model.state_dict(), 'C:/Users/JalynneHEO/OneDrive - 주식회사 투디지트/바탕 화면/ted_eng')