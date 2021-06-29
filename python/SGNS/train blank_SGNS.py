import torch
import pandas as pd
from tqdm import tqdm
import sentencepiece as spm
from model_SGNS import SGNS
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
    for word_num in range(len(sentence)): #쪼개진 단어의 개수를 범위로 지정, word_num = 각 단어의 위치
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
                    output = output.unsqueeze(0) # []처럼 되어있던 크기를 [[]]이렇게 바꾸려고
                    loss = criterion(output, torch.LongTensor([0])) # criterion(output,target)
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
from nltk.tokenize import word_tokenize, sent_tokenize
#%%
urllib.request.urlretrieve("https://raw.githubusercontent.com/GaoleMeng/RNN-and-FFNN-textClassification/master/ted_en-20160408.xml", filename="ted_en-20160408.xml")
#%%
targetXML=open('ted_en-20160408.xml', 'r', encoding='UTF8')
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
sp = spm.SentencePieceProcessor()
vocab_file = './headline.model'
#load the vocab file
sp.Load(vocab_file) #Load
#%%
data = pd.read_excel('C:/Users/JalynneHEO/OneDrive - 주식회사 투디지트/바탕 화면/jalyn_excercise.xlsx', engine = 'openpyxl', dtype = object)
#%%
result = [sp.EncodeAsIds(i) for i in result]

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

torch.save(model.state_dict(), 'C:/Users/JalynneHEO/OneDrive - 주식회사 투디지트/바탕 화면/SGNS_eng')





