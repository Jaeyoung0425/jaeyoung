import torch
import pandas as pd
#from model_other_skipgram import Word2Vec
from tqdm import tqdm
import sentencepiece as spm
from random import randint
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from lxml import etree
from nltk.tokenize import word_tokenize, sent_tokenize
import re
import nltk
import itertools
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.nn.functional as F
from collections import Counter
import random, math
import numpy as np
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


def subsample_frequent_words(corpus): #언제쯤,, 이해가 갈까.. 빈도수가 낮은것은 제외하기
    filtered_corpus = []
    word_counts = dict(Counter(list(itertools.chain.from_iterable(corpus))))
    sum_word_counts = sum(list(word_counts.values()))
    word_counts = {word: word_counts[word]/float(sum_word_counts) for word in word_counts}
    for text in corpus:
        filtered_corpus.append([])
        for word in text:
            if random.random() < (1+math.sqrt(word_counts[word] * 1e3)) * 1e-3 / float(word_counts[word]):
                filtered_corpus[-1].append(word)
    return filtered_corpus


corpus = subsample_frequent_words(result)
vocabulary = set(itertools.chain.from_iterable(corpus))


word_to_index = {w: idx for (idx, w) in enumerate(vocabulary)}
index_to_word = {idx: w for (idx, w) in enumerate(vocabulary)}


def get_batches(context_tuple_list, batch_size=100):
    random.shuffle(context_tuple_list) 
    batches = [] 
    batch_target, batch_context = [], []
    for i in range(len(context_tuple_list)):
        batch_target.append(word_to_index[context_tuple_list[i][0]]) 
        batch_context.append(word_to_index[context_tuple_list[i][1]])
        if (i+1) % batch_size == 0 or i == len(context_tuple_list)-1:
            tensor_target = torch.from_numpy(np.array(batch_target)).long() 
            tensor_context = torch.from_numpy(np.array(batch_context)).long()
            batches.append((tensor_target, tensor_context))
            batch_target, batch_context = [], []
    return batches #tensor들의 list
#%%
print(type([1,2]))
print(type(np.array([1,2])))
print(np.array([1,2])[0])
#%%
context_tuple_list = []
w = 4

for text in tqdm(corpus):
    for i, word in enumerate(text):
        first_context_word_index = max(0,i-w)
        last_context_word_index = min(i+w, len(text))
        for j in range(first_context_word_index, last_context_word_index):
            if i!=j:
                context_tuple_list.append((word, text[j]))

vocabulary_size = len(vocabulary)


loss_function = nn.CrossEntropyLoss()
net = Word2Vec(embedding_size=100, vocab_size=vocabulary_size)
optimizer = optim.Adam(net.parameters())

# while True:
# losses = []
context_tuple_batches = get_batches(context_tuple_list, batch_size=200)
print(len(context_tuple_batches))
for i in tqdm(range(len(context_tuple_batches))):
    net.zero_grad()
    target_tensor, context_tensor = context_tuple_batches[i]
    loss = net(target_tensor, context_tensor)
    loss.backward()
    optimizer.step()

    if i % 1000 == 0:
        print('==========man and woman=============')
        print(torch.cosine_similarity(net.embeddings_target(torch.LongTensor([word_to_index['man']])),
                                      net.embeddings_target(torch.LongTensor([word_to_index['woman']]))))
        print('==========cat and dog =============')
        print(torch.cosine_similarity(net.embeddings_target(torch.LongTensor([word_to_index['cat']])),
                                      net.embeddings_target(torch.LongTensor([word_to_index['dog']]))))
        print('===========paper and rock===========')
        print(torch.cosine_similarity(net.embeddings_target(torch.LongTensor([word_to_index['paper']])),
                                      net.embeddings_target(torch.LongTensor([word_to_index['rock']]))))
        print('=============death and fly===========')
        print(torch.cosine_similarity(net.embeddings_target(torch.LongTensor([word_to_index['death']])),
                                      net.embeddings_target(torch.LongTensor([word_to_index['fly']]))))
        print('==========bee and cat ============')
        print(torch.cosine_similarity(net.embeddings_target(torch.LongTensor([word_to_index['bee']])),
                                      net.embeddings_target(torch.LongTensor([word_to_index['cat']]))))
        
