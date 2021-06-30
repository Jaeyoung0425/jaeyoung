import pandas as pd
import re
from nltk import word_tokenize
from collections import Counter
import itertools
import random
from tqdm import tqdm
import torch
import numpy as np
from rnn_model import RNN_imdb
import sentencepiece as spm
import csv
#%% 
data = pd.read_csv('C:/Users/JalynneHEO/excel/IMDB_Dataset.csv', dtype = object)
index_list = list(data.index)
review_list = [] 
for i in index_list:
    temp_data = data.loc[i,:]
    a = temp_data['review'].replace('<br />','') # br 제거
    a = re.sub(r'[^a-zA-Z ]', '', a.lower()) # 영어, 공백빼고 제거하기
    review_list.append(a)

data['review'] = review_list
result = [word_tokenize(sentence) for sentence in review_list]
word_count = dict(Counter(list(itertools.chain.from_iterable(result))))
word_count = {key: word_count[key] for key in word_count if word_count[key] >= 18}
vocabulary = list(word_count.keys())
vocabulary_len = len(vocabulary)
word_to_index = {w: idx for idx, w in enumerate(vocabulary)}
index_to_word = {idx: w for idx, w in enumerate(vocabulary)}

data['sentiment'] = data['sentiment'].replace(['positive','negative'],[1,0])
dataset = data['review']
target = data['sentiment']

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

# 문장의 최대 길이 : 2262
# 문장의 평균 길이 : 216.546620
# 문장 길이는 단어 400개로 제한 

#%%
print(result_list[0])
#%%
#print(len(dataset[0]))
print(type(vocabulary_len)) 
#%%
print(type(len(vocab_list)))
#%% 
dataset = [i[:400] for i in dataset] 
for _ in range(len(dataset)): 
    if len(dataset[_]) < 400:
        dataset[_] += [vocabulary_len]*(400-len(dataset[_])) # 20992가 남은 부분만큼 채워짐
    else:
        pass

context_tuple_list = []
for _ in range(len(dataset)):
    context_tuple_list.append((dataset[_], target[_])) # ([해당 문장], 긍/부정(1/0)) 이렇게 들어감
#%%
i=[1,2,3,4]
print(len(i)) 
print(len(i)-1) 
#%% 
# context_tuple_list  = [1,2,3,4]
# for i in range(len(context_tuple_list)):
#     if i == len(context_tuple_list) -1:
#         print(context_tuple_list[i]) 

# print(range(len(context_tuple_list))) = range(0,4)
 
#%%
# a = numpy이고
# a[:6:2] = 0~5번에서 2step indexing
#%%
device = ('cuda' if torch.cuda.is_available() else 'cpu')

# context_tuple_list = ([해당 문장], 긍/부정(1/0)) 이렇게 들어감

def get_batches(context_tuple_list, batch_size=100): 
    random.shuffle(context_tuple_list) 
    batches = [] 
    batch_target, batch_context = [], []    # batch_target안에는 문장에 해당하는 [숫자]가 들어감 
    for i in tqdm(range(len(context_tuple_list))): # len(context_tuple_list) = 50000 , range(len(context_tuple_list)) = (0,50000)
        batch_target.append(context_tuple_list[i][0])  # 문장 부분 [숫자]
        batch_context.append(context_tuple_list[i][1]) # 긍/부정 1/0 
        if (i + 1) % batch_size == 0 or i == len(context_tuple_list) - 1:   # -1 은 제일 뒤의 것..
            tensor_target = torch.from_numpy(np.array(batch_target)).long().to(device) #batch_target(숫자)을 numpy로 바꿔줌 = tensor_target
            tensor_context = torch.from_numpy(np.array(batch_context)).long().to(device) #batch_context(1/0)을 numpy로 바꿔줌 = tensor_context
            batches.append((tensor_target, tensor_context)) # batches안에 tensor_target, context 넣어주고 batches return
            batch_target, batch_context = [], []
    return batches 

#%% 
result_list = []
temp_list = []
for _ in range(50000):
    temp_list.append(_)
    if (_+1)%100 == 0: # _ under bar 가 0에서 시작해서 돌면서 99까지 오면 0~99 = 100개, (99+1)/100 = 0이 됨
        print(_, _+1)
        result_list.append(temp_list)
        temp_list = [] 
#%%
print(result_list) 

#%% 

batches = get_batches(context_tuple_list, batch_size = 100) # get_batches 는 def 
model = RNN_imdb(vocabulary_len, batch_size = 100, text_size = 400, input_size = 300, hidden_size = 200, num_layers = 3).to(device)
optimizer = torch.optim.Adam(model.parameters())  # 경사하강법 
criterion = torch.nn.CrossEntropyLoss() 

for _ in range(20):  # 전체 490이 20번 돌게됨 
    for i in tqdm(range(len(batches)-10)): # 전체 batch 500개 중에서 10개는 제외 / 50000 개를 batch_size 100개로 나눴으니까 .. 
        text, sentiment = batches[i]  # batches 안에는 text, sentiment 에 해당하는 tensor가 두개 들어있음
        model.zero_grad() # 한번 학습이 완료되면 grad 를 0으로 초기화, 역전파 하기전에 변화도를 0으로 만들어줌, 미분값을 0으로 초기화
        output = model(text) 
        loss = criterion(output, sentiment) 
        loss.backward()
        optimizer.step() 
        
        # 학습은 490개로 시키고 결과는 495로 확인 
        
        if i % 10 == 0:
            model.eval() 
            test_result = model.forward(batches[495][0]) # 490-500 번째 batch중 아무거나
            test_result = torch.argmax(test_result, dim=1) # torch.argmax = test_result 텐서에 있는 모든 요소의 최대값 <인덱스>를 반환함
            test_answer = batches[495][1] 
            print(torch.sum(test_result == test_answer)) # 개수를 sum 해서 100개니까 sentiment 와 같으면 그거를 정확도로 봄 
            model.train()
        else:
            pass
        
#%% 
print(test_result == test_answer)
#%%
print(batches[495][0])
#%%
print(len(batches)) # batches안에는 텐서가 2개씩 들어가있음 = tensor_target(문장), tensor_context(긍부정)
#%%
print(batches[0])

#%%

#%% sentencepiece code 정리
data = pd.read_csv('C:/Users/JalynneHEO/excel/IMDB_Dataset.csv', dtype = object)
index_list = list(data.index)
review_list = [] 
for i in index_list:
    temp_data = data.loc[i,:]
    a = temp_data['review'].replace('<br />','') # br 제거
    a = re.sub(r'[^a-zA-Z ]', '', a.lower()) # 영어, 공백빼고 제거하기
    review_list.append(a)


with open('imdb_review.txt', 'w', encoding='utf8') as f:
    f.write('\n'.join(data['review']))

spm.SentencePieceTrainer.Train('--input=imdb_review.txt --model_prefix=imdb --vocab_size=10000 --model_type=bpe --max_sentence_length=9999')

vocab_list = pd.read_csv('imdb.vocab', sep='\t', header=None, quoting=csv.QUOTE_NONE)
vocab_list.sample(10)

sp = spm.SentencePieceProcessor()
vocab_file = "imdb.model"
sp.load(vocab_file) 

r = open('./imdb_review.txt', mode='rt', encoding='utf-8')
lines = r.readlines()

data['sentiment'] = data['sentiment'].replace(['positive','negative'],[1,0])
dataset = data['review']
target = data['sentiment'] 

result_list1 = []  
for line in lines: 
    try:
        number = sp.encode_as_ids(line)
        result_list1.append(number)
    except:
        pass

dataset = result_list1

#%%

dataset = [i[:400] for i in dataset] 
for _ in range(len(dataset)): 
    if len(dataset[_]) < 400:
        dataset[_] += [len(vocab_list)]*(400-len(dataset[_])) # 20992가 남은 부분만큼 채워짐
    else:
        pass

context_tuple_list = []
for _ in range(len(dataset)):
    context_tuple_list.append((dataset[_], target[_]))

#%%
device = ('cuda' if torch.cuda.is_available() else 'cpu')

def get_batches(context_tuple_list, batch_size=100): 
    random.shuffle(context_tuple_list) 
    batches = [] 
    batch_target, batch_context = [], []   
    for i in tqdm(range(len(context_tuple_list))): 
        batch_target.append(context_tuple_list[i][0])  
        batch_context.append(context_tuple_list[i][1])
        if (i + 1) % batch_size == 0 or i == len(context_tuple_list) - 1:   
            tensor_target = torch.from_numpy(np.array(batch_target)).long().to(device) 
            tensor_context = torch.from_numpy(np.array(batch_context)).long().to(device) 
            batches.append((tensor_target, tensor_context)) 
            batch_target, batch_context = [], []
    return batches 
#%%

batches = get_batches(context_tuple_list, batch_size = 100) 
model = RNN_imdb(vocabulary_len, batch_size = 100, text_size = 400, input_size = 300, hidden_size = 200, num_layers = 3).to(device)
optimizer = torch.optim.Adam(model.parameters())  
criterion = torch.nn.CrossEntropyLoss() 

for _ in range(20): 
    for i in tqdm(range(len(batches)-10)): 
        text, sentiment = batches[i]  
        model.zero_grad() 
        output = model(text) 
        loss = criterion(output, sentiment) 
        loss.backward()
        optimizer.step() 
        
   
        if i % 10 == 0:
            model.eval() 
            test_result = model.forward(batches[495][0]) 
            test_result = torch.argmax(test_result, dim=1) 
            test_answer = batches[495][1] 
            print(torch.sum(test_result == test_answer))
            model.train()
        else:
            pass
torch.save(model.state_dict(), 'C:/Users/JalynneHEO/OneDrive - 주식회사 투디지트/바탕 화면/RNN')

#%%



