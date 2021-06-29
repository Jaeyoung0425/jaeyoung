# -*- coding: utf-8 -*-
"""
Created on Tue May 18 16:26:20 2021

@author: JalynneHEO
"""
#%%
import torch
import pandas as pd
from tqdm import tqdm
import sentencepiece as spm
from random import randint
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from lxml import etree
from nltk.tokenize import word_tokenize, sent_tokenize
import re
import nltk
from tqdm import tqdm
import itertools
from collections import Counter
import random
import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.nn.functional as F
from model_CBOW import CBOW
#%%
from sklearn.model_selection import train_test_split
#%% 전처리 후 토큰화하기
device = ('cuda' if torch.cuda.is_available() else 'cpu')
targetXML = open('ted_en-20160408.xml', 'r', encoding='UTF8')
target_text = etree.parse(targetXML)

# xml 파일로부터 <content>와 </content> 사이의 내용만 가져온다.
parse_text = '\n'.join(target_text.xpath('//content/text()'))

# 정규 표현식의 sub 모듈을 통해 content 중간에 등장하는 (Audio), (Laughter) 등의 배경음 부분을 제거.
# 해당 코드는 괄호로 구성된 내용을 제거.
content_text = re.sub(r'\([^)]*\)', '', parse_text)

# 입력 코퍼스에 대해서 NLTK를 이용하여 문장 토큰화를 수행.
sent_text = sent_tokenize(content_text) #구문을 문장으로 분리

# 각 문장에 대해서 구두점을 제거하고, 대문자를 소문자로 변환.
normalized_text = [] 
for string in sent_text:
    tokens = re.sub(r"[^a-z0-9]+", " ", string.lower())
    normalized_text.append(tokens)

# 각 문장에 대해서 NLTK를 이용하여 단어 토큰화를 수행.

result = [word_tokenize(sentence) for sentence in normalized_text] 
word_count = dict(Counter(list(itertools.chain.from_iterable(result)))) 
word_count = {key: word_count[key] for key in word_count if word_count[key] >= 5}
vocabulary = list(word_count.keys())
vocabulary_len = len(vocabulary)

word_to_index = {w: idx for idx, w in enumerate(vocabulary)} 
index_to_word = {idx: w for idx, w in enumerate(vocabulary)} 

for i in range(len(result)):
    result[i] = [word_to_index[j] for j in result[i] if j in vocabulary] 
    #j는 result[i]안에 있는 원소인데 j가 vocab안에 있으면 wordtoindex[j]를 result[i]리스트 안에 넣기

w = 2
context_tuple_list = []
for text in tqdm(result):
    for i, word in enumerate(text):
        first_context_word_index = max(0, i - w)
        last_context_word_index = min(i + w + 1, len(text))
        temp_list = []
        # target_list = [0]*vocabulary_len
        # target_list[word] = 1
        for j in range(first_context_word_index, last_context_word_index):
            if i != j:
                temp_list.append(text[j])
                
        if len(temp_list) < 4:
            temp_list = temp_list + [vocabulary_len] * (4 - len(temp_list))
        context_tuple_list.append((word, temp_list))

#%%
print(word_to_index)

#%%
model=CBOW(vocabulary_len,100)
#%%
model.load_state_dict(torch.load('C:/Users/JalynneHEO/model_cbow.bin',map_location=device))
#%%

#print('==========huge and big=============')
a=torch.cosine_similarity(model.Embedding(torch.LongTensor([word_to_index['huge']]).to(device)),
                              model.Embedding(torch.LongTensor([word_to_index['big']]).to(device)))

#print('==========cat and dog =============')
b=torch.cosine_similarity(model.Embedding(torch.LongTensor([word_to_index['cat']]).to(device)),
                              model.Embedding(torch.LongTensor([word_to_index['dog']]).to(device)))

#print('===========paper and rock===========')
c=torch.cosine_similarity(model.Embedding(torch.LongTensor([word_to_index['paper']]).to(device)),
                              model.Embedding(torch.LongTensor([word_to_index['rock']]).to(device)))

#%%
print(type(model.Embedding(torch.LongTensor([word_to_index['paper']]))))
#%%
a_list = [a, b, c]
#%%
print(type(a))
#%%
print(a_list[1].item())
#type = float
#%%
print(type(vocabulary))
#%%
print(model.Embedding(torch.LongTensor([word_to_index['huge']])))

#%%
print([word_to_index['huge']])
print([index_to_word[266]])
#%%
import operator 
#%%

def similar_words(vocab): 
    # comparing cosine similarity with the input word and 21613 words
    sim_dic = {}
    for i in range(len(vocabulary)): 
        similarity = torch.cosine_similarity(model.Embedding(torch.LongTensor([word_to_index[vocabulary[i]]]).to(device)), 
                                                 model.Embedding(torch.LongTensor([word_to_index[vocab]]).to(device))) 
                                        
        sim_dic[vocabulary[i]] = similarity.item() 
    
    order_dic = {k: v for k, v in sorted(sim_dic.items(), key=lambda item: item[1])}
    desc_dic = sorted(sim_dic.items(),key=operator.itemgetter(1),reverse=True)
    desc_dic = desc_dic[:6] # list로 형태가 바뀜
    
    return desc_dic


#%%
a = similar_words('man')
#%%
print(a)
#%%
a = 3
#%%
def test():
    global a
    a = 1
test()
#%%
print(a) 
#%%
print(vocabulary[:10])
#%% 
print(sim_dic)
#%%

print(similarity)
#%%
print(word_to_index['libra'])
#%%
# sorted(sim_dic, key= lambda x: dict[x])

#%%
print(sim_dic)
#   
    # Then return top 5 words ( using descending order ..? ) 
    # similar = [similar_tensor[i].item() for i in similar_tensor] # 아니야 ... 
#%%
# dartmouth college, nh , us 함승현
# 박이령  dasliebeich7
# 이예령  ery270
#%%
vocab = 'cat'
print(model.Embedding(torch.LongTensor([word_to_index[vocab]]).to(device)))
#%%

for i in range(len(vocabulary)): 
    print(vocabulary[:7])
#%%
vocab_size = len(index_to_word)
print(vocab_size)
#%%
print(len(index_to_word))
print(len(word_to_index))
# count = 0
# for j in (-1*similarity).argsort():
#     if index_to_word[j] == vocab:
#         continue
#     print(' %s: %s' % (index_to_word[j],similarity[j]))
    
#     count += 1
#     if count >= top:
#         return 

#%%
import pymysql
#%%
def get_maria_connection():
    conn = None
    for _ in range(3):
        try:
            conn = pymysql.connect(host='database-1.c1pvc7savwst.ap-northeast-2.rds.amazonaws.com',
                                   user='nsuser',
                                   password='nsuser))))',
                                   db='NewsSalad_dev_01',
                                   charset='utf8')
            return conn

 

        except:
            pass
    return conn
#%%
conn = get_maria_connection()
#%%
with conn.cursor(pymysql.cursors.DictCursor) as curs:
    query = '''SELECT symbol, title 
    FROM TB_YAHOO_NEWS_US'''
    curs.execute(query)
    data = curs.fetchall()
    # data = list안에 dict이 들어가 있는 형태
#%%
print(data[:5])
#%%
print(type(data)) #

#%% 
#%% 종목코드 & 타이틀 리스트
code_list = []
for i in range(len(data)):
    code_list.append(list(data[i].values())[0]) 
    
title_list = []
for i in range(len(data)):
    title_list.append(list(data[i].values())[1])  
#%%
print(title_list)
#%%
print(code_list)
#%%
print(type(title_list))

#%% 
normalized_title = []
for string in title_list:
    tokens = re.sub(r"[^a-z0-9]+", " ", string.lower())
    normalized_title.append(tokens)
print(normalized_title)
#%%
result = [word_tokenize(sentence) for sentence in normalized_title]
#%%
print(len(result))
#%%
print(result[0])
#%%
# sum_ = []
# for i in range(len(result)):
#     for j in range(len(result[i])):
#         try:
#             sum_ += model.Embedding(torch.LongTensor([word_to_index[result[i][j]]]))
#         except:
#             pass
#%%

#%% 종목코드 & 정규화 끝낸거 합쳐서 리스트로 만들기
list3 = list(zip(code_list,result))
#%%
print(list3[12][1][3]) 
#%%
print(list3[0][1][1])
#%%
print(len(list3))
#%%
print(list3)
#%%
title_list = []
for i in range(len(title_list)):
    title = re.sub(r'\([^)]*\)', '', title_list[i])
    title_list.append(title)
    print(title_list)
#%% embedding sample 
a=model.Embedding(torch.LongTensor([word_to_index['huge']]))
b=model.Embedding(torch.LongTensor([word_to_index['big']]))
print(a)
print(b)
print(b+a)
#%%  

sum_tensor = []
for i in range(len(list3)):
    temp_tensor = []
    for j in range(len(list3[i][1])):
        try:
            temp_tensor += list(model.Embedding(torch.LongTensor([word_to_index[list3[i][1][j]]])).to(device))
        except Exception as e:
            pass
        
    temp_tensor = sum(temp_tensor)
    sum_tensor.append(temp_tensor)
    

#%%
print(sum_tensor)  
#%%
print(code_list[:10])
#%% 
list4 = list(zip(code_list,sum_tensor)) 
#%%
data = pd.DataFrame(list4, columns = ['symbol', 'tensor'])
#%%
print(data)
#%%
print(len(list4)) # =10320
#%% 
print(type(list4[0][1])) # = torch.Tensor
#%%
print(list4)
#%% 코사인 유사도 구하기 
# ??   
print(torch.cosine_similarity(list4[0][1].unsqueeze(0), list4[1][1].unsqueeze(0)))
      
#%%
print(list4[:10])
#%%
print(len(list4))
#%%
print(len(data))
#%%
import time
#%% 
cnt= 0
for unique_symbol in list(data['symbol'].unique()):
    #print(unique_symbol)
    
    mask = (data['symbol'] == unique_symbol)  
    temp_data = data.loc[mask,:]
    # print(temp_data)
    time.sleep(5)
    
    index_list = list(temp_data.index)
    for i in range(len(index_list)):
        for j in range(i+1, len(index_list)):
            sim = torch.cosine_similarity(data['tensor'][i].unsqueeze(0),data['tensor'][j].unsqueeze(0))
            print(index_list[i], index_list[j], data['symbol'][i], sim) 
            # if sim.item() > 0.7: 
            #     pass
            # else:
            #     print(index_list[i], index_list[j], sim) 

            
                #print(sim.item())
#%%
print(len(data['symbol'].unique()))
#%% 
print(len(index_list))
#%%
print(temp_data)
#%%
# type(sim.item()) = float
#%%
print(torch.cosine_similarity(data['tensor'][1].unsqueeze(0),data['tensor'][2].unsqueeze(0)))

#%%
print(data['tensor'])

#%%
a += list(b)
#%%
a += list(b)
#%%
print(sum(a))
#%%
print(a)
#%%
print(None + b)

#%%
print(a)
 
#%%
print(sum([1,2]))
#%%
print(len(sum_tensor))  
#%%
import pandas as pd
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
import re
import urllib.request
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
#%%
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename="ratings_train.txt")
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt", filename="ratings_test.txt")
#%%
train_data = pd.read_table('ratings_train.txt')
test_data = pd.read_table('ratings_test.txt')
#%%
print('훈련용 리뷰 개수 :',len(train_data))
#%%
print(train_data)
#%%
print(train_data[:5])
#%%
print((train_data['document'].nunique(), train_data['label'].nunique()))
#%% 중복 샘플 제거하기
train_data.drop_duplicates(subset=['document'], inplace=True)
#%%
print(len(train_data))
#%%
print(train_data['label'].value_counts().plot(kind='bar'))
#%%
print(train_data.groupby('label').size().reset_index(name = 'count'))
#%% null값을 가진 샘플이 있는지 확인하기
print(train_data.isnull().values.any())
#%% null값이 존재하는 행 제거
train_data = train_data.dropna(how = 'any')
#%% null값 없어짐
print(train_data.isnull().values.any())
#%%
print(len(train_data))
#%% 
train_data['document'] = train_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
#%%
print(train_data[:5])
#%%
train_data['document'] = train_data['document'].str.replace('^ +', "")
train_data['document'].replace('', np.nan, inplace=True)
print(train_data.isnull().sum())
#%%
print(train_data.loc[train_data.document.isnull()][:5])
#%%
train_data = train_data.dropna(how = 'any')
print(len(train_data))

#%% test data 전처리하기
test_data.drop_duplicates(subset = ['document'], inplace=True) # document 열에서 중복인 내용이 있다면 중복 제거
test_data['document'] = test_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","") # 정규 표현식 수행
test_data['document'] = test_data['document'].str.replace('^ +', "") # 공백은 empty 값으로 변경
test_data['document'].replace('', np.nan, inplace=True) # 공백은 Null 값으로 변경
test_data = test_data.dropna(how='any') # Null 값 제거
print('전처리 후 테스트용 샘플의 개수 :',len(test_data))
#%%
okt = Okt()
okt.morphs('와 이런 것도 영화라고 차라리 뮤직비디오를 만드는 게 나을 뻔', stem = True)
# stem = True / 일정수준의 정규화를 수행해줌
# cnt = count 변수
#%%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dropout, Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
#%%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn import preprocessing
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
import urllib.request
#%% numpy 배열 파일 = npy
urllib.request.urlretrieve("https://github.com/ajinkyaT/CNN_Intent_Classification/raw/master/data/train_text.npy", filename="train_text.npy")
urllib.request.urlretrieve("https://github.com/ajinkyaT/CNN_Intent_Classification/raw/master/data/test_text.npy", filename="test_text.npy")
urllib.request.urlretrieve("https://github.com/ajinkyaT/CNN_Intent_Classification/raw/master/data/train_label.npy", filename="train_label.npy")
urllib.request.urlretrieve("https://github.com/ajinkyaT/CNN_Intent_Classification/raw/master/data/test_label.npy", filename="test_label.npy")
#%% 
old = np.load
np.load = lambda *a,**k: old(*a,allow_pickle=True,**k)
#%% url로 받은 데이터를 리스트로 저장하기

intent_train = np.load(open('train_text.npy', 'rb')).tolist()
label_train = np.load(open('train_label.npy', 'rb')).tolist()
intent_test = np.load(open('test_text.npy', 'rb')).tolist()
label_test = np.load(open('test_label.npy', 'rb')).tolist()
#%%
print(label_train)
#%% 레이블에 고유한 정수 부여하기 
idx_encode = preprocessing.LabelEncoder()
idx_encode.fit(label_train)
#%%  
label_train = idx_encode.transform(label_train) # 주어진 고유한 정수로 변환
label_test = idx_encode.transform(label_test) # 고유한 정수로 변환
#%%
label_idx = dict(zip(list(idx_encode.classes_), idx_encode.transform(list(idx_encode.classes_))))
print(label_idx)
#%%
print(intent_test[:5])
print(label_test[:5])

# 토큰화 수행, 단어 집합 만들기, 정수 인코딩 수행
#%%
tokenizer = Tokenizer()
tokenizer.fit_on_texts(intent_train) # tokenizer.fit_on_texts(sentences) = 빈도수를 기준으로 단어 집합을 생성함
sequences = tokenizer.texts_to_sequences(intent_train) 
print(sequences[:5]) # 상위 5개 샘플 출력
#%%
print(sequences)
#%%
word_index = tokenizer.word_index
vocab_size = len(word_index) + 1
print('단어 집합(Vocabulary)의 크기 :',vocab_size)
#%%
print(word_index)
#%%
max_len = 35
intent_train = pad_sequences(sequences, maxlen = max_len)
label_train = to_categorical(np.asarray(label_train))
print('전체 데이터의 크기(shape):', intent_train.shape)
print('레이블 데이터의 크기(shape):', label_train.shape)

#%%
indices = np.arange(intent_train.shape[0])
np.random.shuffle(indices)
print(indices)
#%%
# 시퀀스 = 연속형 자료형 ex) 튜플, 리스트
# 시퀀스 데이터들은 for in 구문을 사용하거나 인덱스 연산자를 사용하여 전체 데이터를 순회할 수 있음
#%%

# 문장의 길이가 다를 때 패딩, ex) 나머지 칸들에 0 집어넣기
# 태깅 작업 = 대표적인 시퀀스 레이블링 작업
#%%
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import imdb
#%%
data = pd.read_csv('C:/Users/JalynneHEO/IMDB_Dataset.csv', dtype = object) 
#%%
print(data['sentiment'])
#%%
print(data['review'])

#%% 
print(data.groupby('sentiment').size().reset_index(name='count'))
# positive / negative 각각 25000개씩으로 even함
#%%
re.sub(r'[^a-zA-Z ]', '', text)
#%%
review_list=list(data['review'])
print(review_list)
#%%
index_list = list(data.index)
review_list = [] # normalized_text 가 여기에서 review_list
for i in index_list:
    temp_data = data.loc[i,:]
    a = temp_data['review'].replace('<br />','') # br 제거
    a = re.sub(r'[^a-zA-Z ]', '', a.lower()) # 영어, 공백빼고 제거하기
    review_list.append(a)
    
data['review'] = review_list
#%%
print(data['review'][0])
#%%

#%%
result = [word_tokenize(sentence) for sentence in review_list]
word_count = dict(Counter(list(itertools.chain.from_iterable(result))))
word_count = {key: word_count[key] for key in word_count if word_count[key] >= 18}
vocabulary = list(word_count.keys())
vocabulary_len = len(vocabulary)

word_to_index = {w: idx for idx, w in enumerate(vocabulary)} 
index_to_word = {idx: w for idx, w in enumerate(vocabulary)} 

#%%
print(word_count)
#%%
word_sorted = sorted(word_count.items(), key = lambda x:x[1], reverse = True)
print(word_sorted)
#%% 20992개 
print(vocabulary_len)
#%%
print(word_to_index)
#%%
print([word_to_index['fly']])
#%%
data['sentiment'] = data['sentiment'].replace(['positive','negative'],[1,0])
#%%
dataset = data['review']
target = data['sentiment']
#%% 0.2/0.8(=40,000개)
x_train, x_test, y_train, y_test = train_test_split(dataset, target, train_size=0.8, shuffle = True, random_state = 1)

#%% train = 40000개
print(len(x_train))

#%%
print(x_train[0])

#%%
import pandas as pd
#%%
data = pd.read_csv('C:/Users/JalynneHEO/IMDB_Dataset.csv', dtype = object)
#%%
index_list = list(data.index)
review_list = [] # normalized_text 가 여기에서 review_list
for i in index_list:
    temp_data = data.loc[i,:]
    a = temp_data['review'].replace('<br />','') # br 제거
    a = re.sub(r'[^a-zA-Z ]', '', a.lower()) # 영어, 공백빼고 제거하기
    review_list.append(a)
    
data['review'] = review_list
#%%
result = [word_tokenize(sentence) for sentence in review_list]
word_count = dict(Counter(list(itertools.chain.from_iterable(result))))
word_count = {key: word_count[key] for key in word_count if word_count[key] >= 18}
vocabulary = list(word_count.keys())
vocabulary_len = len(vocabulary)

word_to_index = {w: idx for idx, w in enumerate(vocabulary)} 
index_to_word = {idx: w for idx, w in enumerate(vocabulary)} 
#%%
data['sentiment'] = data['sentiment'].replace(['positive','negative'],[1,0])
#%%
dataset = data['review']
target = data['sentiment']
#%% 
x_train, x_test, y_train, y_test = train_test_split(dataset, target, train_size=0.8, shuffle = True, random_state = 1)
#%%
#1 dataset에 들어있는 리뷰 정보를 숫자로 바꾸기 O
#2 사전에 없는 단어는 지우기 O
#3 문장 길이 분포 알아보기 O
#%%
print(list(dataset))

#%%
print(dataset[:10])
#%%
result_list = []
for sentence in list(dataset):
    numbering = []
    for s in word_tokenize(sentence): 
        try:
            index = word_to_index[s]
            numbering.append(index)
            
        except:
            pass
    result_list.append(numbering)

dataset = result_list     

#%%

print('문장의 최대 길이 : %d' % max(len(l) for l in x_train))
print('문장의 평균 길이 : %f' % (sum(map(len, x_train))/len(x_train)))
plt.hist([len(s) for s in x_train], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()

#%%
print(x_train)

#%%
syn_data = pd.read_csv('C:/Users/JalynneHEO/krx_list_synonym_new_2021_0512.csv', dtype = object)
#%%
syn_df = pd.read_excel('C:/Users/JalynneHEO/stock_synonym.xlsx', engine='openpyxl', dtype=object)
#%%
print(syn_df) 
#%%  
print(syn_df['code'])
print(syn_df['new_syn'])
#%%
code_list = list(syn_df['code'])
new_syn_list = list(syn_df['new_syn'])
#%%
new_syn_list = [i.split(',') for i in new_syn_list]
#%%
len_list = [len(i) for i in new_syn_list]

#%%
print(code_list)
print(new_syn_list)

#%%
index_list = list(syn_df.index)
result_list = []

for temp_index in index_list:
    try:
        temp_data = syn_df.loc[temp_index, :]
        r = temp_data['new_syn'].strip().replace(', ',',')
        result_list.append(r)
    except:
        result_list.append(temp_data['new_syn'])

syn_df['new_syn']=result_list
#%%
print(list(syn_df['new_syn']))
#%%
print(result_list)
        
#%%
print(dic)
#%%
df.to_csv('C:/Users/JalynneHEO/hahaJY.csv', index = False)
#%% '단축코드' + '신규 동의어' 
syn_df = syn_df.dropna(subset=['new_syn'])
print(syn_df)
#%%
print(syn_df['code'])
#%%
print(syn_df['new_syn'])
#%%
new_df = syn_df['code'].map(str)+","+syn_df['new_syn']
#%%
print(new_df)
#%%
print(new_df).columns
#%%
print(syn_df.columns)
#%% 
new_df.to_csv("C:/Users/JalynneHEO/new_syn.csv", index=False, sep = '|')
#%%
print(syn_df.columns)
#%%
print(syn_df.drop('syn', axis=1, inplace=True))
#%%
code	kor 	s_kor	eng 	syn 	new_syn	etc  
#%%
syn_df.drop("syn")
#%%







