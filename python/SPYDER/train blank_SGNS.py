import torch
import pandas as pd
from tqdm import tqdm
import sentencepiece as spm
from model_blank_SGNS_batch import SGNS #SGNS 만든거 가져오기
from torch import nn
#%%
sp = spm.SentencePieceProcessor()
vocab_file = './headline.model'
#load the vocab file
sp.Load(vocab_file) #Load

data = pd.read_excel('C:/Users/JalynneHEO/OneDrive - 주식회사 투디지트/바탕 화면/jalyn_excercise.xlsx', engine = 'openpyxl', dtype = object)
#%%
#lower given title
data['title'] = data['title'].str.lower() 
title_list = data['title'].tolist() 

#encode each sentence
title_list = [sp.EncodeAsIds(i) for i in title_list] # EncodeAsIds = 문장을 입력하면 sub_word sequence로 변환함
#EncodeAsIds = 문장을 입력하면 각 단어에 해당하는 번호를 리스트의 형태로 반환해줌

model = SGNS(sp.GetPieceSize(), 50)

#define loss, loss = CrossEntropyLoss
criterion = torch.nn.CrossEntropyLoss #nn.CrossEntropyLoss()

#define optimizer, adam and learning rate = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=0.01) 
for sentence in tqdm(title_list): #sentence = title_list안에 들어있는 쪼개진 문장들
    for word_num in range(len(sentence)): #쪼개진 단어의 개수를 범위로 지정, word_num = 쪼개진 단어 개수
        # for words in window size 2, label 1 and learn
        for _ in [-2,-1,1,2]: #중심 단어 기준으로 주변단어 가져오려고
            if word_num + _ < 0: #word_num = 중심단어의 위치 (모든 단어가 한번씩 중심 단어가 되어야함)
                pass #word_num = 리스트에서의 위치
            elif word_num + _ >= len(sentence): #word_num은 0부터 시작해서 len(sentence)보다 짧음
                pass
            else:
                #input two number
                output = model.forward(sentence[word_num],sentence[word_num +_]) # number_1, number_2              
                
                if output is None:
                    pass
                else:
                    output = output.unsqueeze(0)
                    loss = criterion(output, torch.LongTensor([1])) #criterion(예측, 정답)
                    #criterion = 기준, 어떤 loss를 줄일 것인지에 대해서 명시함
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        # get random 5 words for each center word
        random_words = torch.randint(0, sp.GetPieceSize(), (5,)) #0부터 sp.GetPieceSize()사이의 랜덤한 자연수, (5,) = 텐서 크기
        
        #(5,1) = [[1,2,3,4,5]]
        #(5,0) = [1,2,3,4,5]
        
        #random_words = tensor 텐서임 텐서!! 
        for _ in range(5): # 0~4까지 범위로 설정한 이유 = 랜덤한 주변 단어 5개 가져오려고
            # if random word is in neighbor, pass 
            if random_words[_].item() in sentence[max(0,word_num -2):word_num + 2]: 
                pass
            
            else: 
                output = model.forward(sentence[word_num],random_words[_].item()) 
                
                if output is None: 
                    pass
                else:
                    output = output.unsqueeze(0)
                    loss = criterion(output, torch.LongTensor([0])) 
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

torch.save(model.state_dict(), 'C:/Users/JalynneHEO/OneDrive - 주식회사 투디지트/바탕 화면/SGNS')
#%%
print(torch.LongTensor([0]))
print(torch.LongTensor([1]))
#%%
sen = ['a','b','c','d','e']
print(len(sen)) 
#%%

data['title'] = data['title'].str.lower()
title_list = data['title'].tolist()

title_list = [sp.EncodeAsIds(i) for i in title_list]
#%%
model = SGNS(sp.GetPieceSize(), 50)
criterion = torch.nn.CrossEntropyLoss

optimizer = torch.optim.Adam(model.parameters(), lr=0.01) 
for sentence in tqdm(title_list): 
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

torch.save(model.state_dict(), 'C:/Users/JalynneHEO/OneDrive - 주식회사 투디지트/바탕 화면/SGNS')
#%%
import urllib.request
import zipfile
from lxml import etree
import re
import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
#%%
urllib.request.urlretrieve("https://raw.githubusercontent.com/GaoleMeng/RNN-and-FFNN-textClassification/master/ted_en-20160408.xml", filename="ted_en-20160408.txt")
#%%
targetXML=open('ted_en-20160408.txt', 'r', encoding='UTF8')
target_text = etree.parse(targetXML)

# xml 파일로부터 <content>와 </content> 사이의 내용만 가져온다.
parse_text = '\n'.join(target_text.xpath('//content/text()'))

# 정규 표현식의 sub 모듈을 통해 content 중간에 등장하는 (Audio), (Laughter) 등의 배경음 부분을 제거.
# 해당 코드는 괄호로 구성된 내용을 제거.
content_text = re.sub(r'\([^)]*\)', '', parse_text)

# 입력 코퍼스에 대해서 NLTK를 이용하여 문장 토큰화를 수행.
sent_text = sent_tokenize(content_text)

# 각 문장에 대해서 구두점을 제거하고, 대문자를 소문자로 변환.
normalized_text = []
for string in sent_text:
     tokens = re.sub(r"[^a-z0-9]+", " ", string.lower())
     normalized_text.append(tokens)

# 각 문장에 대해서 NLTK를 이용하여 단어 토큰화를 수행.
result = [word_tokenize(sentence) for sentence in normalized_text]
#%%
print(result)
#%%
import sentencepiece as spm
import pandas as pd
import urllib.request
import csv
#%%
sp = spm.SentencePieceProcessor()
vocab_file = './ted.model'
#load the vocab file
sp.Load(vocab_file) #Load
#%%
data = pd.read_excel('C:/Users/JalynneHEO/OneDrive - 주식회사 투디지트/바탕 화면/jalyn_excercise.xlsx', engine = 'openpyxl', dtype = object)
#%%
with open('ted_eng.txt') as data:
   lines = data.readlines()

#%% 
result = [word_tokenize(sentence) for sentence in normalized_text]
#print(result)
#%%
result = [sp.EncodeAsIds(i) for i in lines]
#%%

#%%


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
#%%
import torch
import torch.nn as nn
import torch.nn.functional as F

#%%
import urllib.request
import pandas as pd
#%%

ted = pd.read_csv('C:/Users/JalynneHEO/Downloads/ted_en-20160408.csv', engine = 'python-fwf', dtype = object)
#%%
print(ted)

#%% 
# result = [sp.EncodeAsIds(i) for i in result]

model = SGNS(sp.GetPieceSize(), 50)
#%%
model.load_state_dict(torch.load('C:/Users/JalynneHEO/model/model.bin'))

#%%
sp = spm.SentencePieceProcessor()
vocab_file = './ted.model'
#load the vocab file
sp.Load(vocab_file) #Load

#%%
print(sp.EncodeAsIds('dog'))
print(sp.Decode([123]))
print(sp.Decode([2948]))
#%%

#%%
print(dir(model))
#%%
# model.linear1 = lookup table
#%%
import pandas as pd
import sentencepiece as spm
import urllib.request
import csv
#%%   
dog_split = sp.Decode([123])+sp.Decode([2948]) 
print(dog_split) # = dog, (type = string) 

#%%
print(sp.EncodeAsIds('dog'))
#%% dog  
dog_list = sp.EncodeAsIds('dog') 
tensor_dog = torch.Tensor([0]*50)
for _ in dog_list:
    temp_tensor = torch.Tensor([0]*3000)
    temp_tensor[_] = 1
    tensor_dog += model.linear1(temp_tensor)
print(tensor_dog)
#%%
print(sp.EncodeAsIds('cat'))
print(sp.Decode([31]))
print(sp.Decode([24]))
#%%
cat_split = sp.Decode([31])+sp.Decode([24]) 
print(cat_split) # = dog, (type = string) 
#%%
cat_list = sp.EncodeAsIds('cat') 
tensor_cat = torch.Tensor([0]*50)
for _ in cat_list:
    temp_tensor = torch.Tensor([0]*3000)
    temp_tensor[_] = 1
    tensor_cat += model.linear1(temp_tensor)
print(tensor_cat)
#%%
print(torch.cosine_similarity(tensor_dog, tensor_cat, dim=0))
#%%
print(sp.EncodeAsIds('man'))
print(sp.Decode([377]))
#%%
man = sp.EncodeAsIds('man')
tensor_man = torch.Tensor([0]*50)
for _ in man:
    temp_tensor = torch.Tensor([0]*3000)
    temp_tensor[_] = 1
    tensor_man += model.linear1(temp_tensor)
print(tensor_man)
#%%
print(sp.EncodeAsIds('woman'))

print(sp.Decode([1760]))
#%%
woman = sp.EncodeAsIds('woman')
tensor_woman = torch.Tensor([0]*50)
for _ in woman:
    temp_tensor = torch.Tensor([0]*3000)
    temp_tensor[_] = 1
    tensor_woman += model.linear1(temp_tensor)
print(tensor_woman)
#%%
boy = sp.EncodeAsIds('boy')
tensor_boy = torch.Tensor([0]*50)
for _ in boy:
    temp_tensor = torch.Tensor([0]*3000)
    temp_tensor[_] = 1
    tensor_boy += model.linear1(temp_tensor)
print(tensor_boy)
#%%
print(sp.EncodeAsIds('girl'))

print(sp.Decode([1549]))
#%%
girl = sp.EncodeAsIds('girl')
tensor_girl = torch.Tensor([0]*50) #W_len = 50
for _ in girl: 
    tensor_girl += model.embedding2(torch.tensor(_)) 
print(tensor_girl) 
#%%
woman = sp.EncodeAsIds('woman')
tensor_woman = torch.Tensor([0]*50) #W_len = 50
for _ in woman:
    tensor_woman += model.embedding2(torch.tensor(_)) 
print(tensor_woman) 
#%%
boy = sp.EncodeAsIds('boy')
tensor_boy = torch.Tensor([0]*50)
for _ in boy:
    tensor_boy += model.embedding2(torch.tensor(_))
print(tensor_boy)
#%%
man = sp.EncodeAsIds('man')
tensor_man = torch.Tensor([0]*50)
for _ in man:
    tensor_man += model.embedding2(torch.tensor(_))
print(tensor_man)
#%%
cat = sp.EncodeAsIds('cat')
tensor_cat = torch.Tensor([0]*50)
for _ in cat:
    tensor_cat += model.embedding2(torch.tensor(_))
print(tensor_cat)
#%%
kid = sp.EncodeAsIds('kid')
tensor_kid = torch.Tensor([0]*50)
for _ in kid:
    tensor_kid += model.embedding2(torch.tensor(_))
print(tensor_kid)
#%%
children = sp.EncodeAsIds('children')
tensor_children = torch.Tensor([0]*50)
for _ in children:
    tensor_children += model.embedding2(torch.tensor(_))
print(tensor_children)
#%%
print(torch.cosine_similarity(tensor_kid, tensor_children, dim=0))

#%%
dog = sp.EncodeAsIds('dog')
tensor_dog = torch.Tensor([0]*50)
for _ in dog:
    tensor_dog += model.embedding2(torch.tensor(_))
print(tensor_dog)
#%%
tooth = sp.EncodeAsIds('tooth')
tensor_tooth = torch.Tensor([0]*50)
for _ in tooth:
    tensor_tooth += model.embedding1(torch.tensor(_))
print(tensor_tooth)
#%%
brush = sp.EncodeAsIds('brush')
tensor_brush = torch.Tensor([0]*50)
for _ in brush:
    tensor_brush += model.embedding1(torch.tensor(_))
print(tensor_brush)
#%%
eat = sp.EncodeAsIds('eat')
tensor_eat = torch.Tensor([0]*50)
for _ in eat:
    tensor_eat += model.embedding1(torch.tensor(_))
print(tensor_eat)
#%%
apple = sp.EncodeAsIds('apple')
tensor_apple = torch.Tensor([0]*50)
for _ in apple:
    tensor_apple += model.embedding1(torch.tensor(_))
print(tensor_apple)
#%%
banana = sp.EncodeAsIds('banana')
tensor_banana = torch.Tensor([0]*50)
for _ in banana:
    tensor_banana += model.embedding1(torch.tensor(_))
print(tensor_banana)
#%%
job = sp.EncodeAsIds('job')
tensor_job = torch.Tensor([0]*50)
for _ in job:
    tensor_job += model.embedding1(torch.tensor(_))
print(tensor_job)
#%%
training = sp.EncodeAsIds('training')
tensor_training = torch.Tensor([0]*50)
for _ in training:
    tensor_training += model.embedding1(torch.tensor(_))
print(tensor_training)
#%%
print(torch.cosine_similarity(tensor_job, tensor_training, dim=0))

#%%
print(torch.cosine_similarity(tensor_tooth, tensor_brush, dim=0))

#%%
print(torch.cosine_similarity(tensor_man, tensor_cat, dim=0))
print(torch.cosine_similarity(tensor_dog, tensor_cat, dim=0))
print(torch.cosine_similarity(tensor_man, tensor_dog, dim=0))
print(torch.cosine_similarity(tensor_man, tensor_woman, dim=0))
print(torch.cosine_similarity(tensor_man, tensor_boy, dim=0)) 
print(torch.cosine_similarity(tensor_woman, tensor_girl, dim=0))
print(torch.cosine_similarity(tensor_man, tensor_girl, dim=0))
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
#임베딩의 입력 = 단어의 인덱스 = 정수 
girl = sp.EncodeAsIds('girl')

for _ in girl: 
    model.embedding1(_)


#%%
print(torch.cosine_similarity(tensor_man, tensor_cat, dim=0))
print(torch.cosine_similarity(tensor_dog, tensor_cat, dim=0))
print(torch.cosine_similarity(tensor_man, tensor_dog, dim=0))
print(torch.cosine_similarity(tensor_man, tensor_woman, dim=0))
print(torch.cosine_similarity(tensor_man, tensor_boy, dim=0)) 
print(torch.cosine_similarity(tensor_woman, tensor_girl, dim=0))
print(torch.cosine_similarity(tensor_man, tensor_girl, dim=0))
#%%
vocab_list = pd.read_csv('ted.model', sep='\t', header=None, quoting=csv.QUOTE_NONE)
vocab_list.sample(10)
print(len(vocab_list)) #3000개

#%%
# using gensim package 
from gensim.models import Word2Vec, KeyedVectors
#%% skipgram
skipgram = Word2Vec(sentences=result, vector_size=100, window=5, min_count=5, workers=4, sg=1)
#%%
skipgram_result = skipgram.wv.most_similar("man")
print(skipgram_result)
#%% cbow
cbow = Word2Vec(sentences=result, vector_size=100, window=5, min_count=5, workers=4, sg=0)
#%%
cbow_result = cbow.wv.most_similar("man")
print(cbow_result)
#%%







