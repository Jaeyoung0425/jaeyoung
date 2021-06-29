import torch
import pandas as pd
from model_blank_SGNS_batch import SGNS
from tqdm import tqdm
import sentencepiece as spm
from random import randint
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
# %%
sp = spm.SentencePieceProcessor()
vocab_file = './ted.model'
# load the vocab file
sp.Load(vocab_file)
r = open('./ted_eng.txt', mode='rt', encoding='utf-8')
line = r.readlines()
# %% title_list = ted_eng에서 한줄씩 읽어온 것
# #lower given title
title_list = [i.lower() for i in line]
# #encode each sentence
title_list = [sp.EncodeAsIds(i) for i in title_list]
# %%
model = SGNS(sp.GetPieceSize(), 50) #단어개수, 차원
#%%
# model.to("cuda")
# define loss, loss = CrossEntropyLoss 

criterion = torch.nn.CrossEntropyLoss()
#%%
# define optimizer, adam and learning rate = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# center_list : the list of center words in sentence 
# neighbor_list : the list of neighbor words in sentence 
# rand_list : the list of random words in sentence 
center_list = [] 
neighbor_list = [] # <- temp_list = title[word_num + _]
rand_list = [] 

for title in tqdm(title_list): 
    for word_num in range(len(title)): #word_num = 단어 각각의 위치 (잠재적 중심단어)
        temp_list = []
        for _ in [-2, -1, 1, 2]:
            if (word_num + _ < 0) or (word_num + _ >= len(title)):
                pass 
            else:
                temp_list.append(title[word_num + _])  #각 문장에서 0보다 크면서 문장 길이를 넘지 않는 위치에 해당하는 단어를 temp_list에 넣기

        center_list.append([title[word_num]])  
        neighbor_list.append(temp_list)  # temp_list = title[word_num + _] = 주변단어 
        
        rand = [randint(0, sp.GetPieceSize() - 1) for i in range(6)] #0~5 = 6개 # sp.GetPieceSize() = 3000(0~2999), 0부터 세니까 1을 빼줘야 함, 3000이 뽑히면 안됨!!
        rand = set(rand) # set()함수 = 중복제거 

        rand = rand - set(temp_list)  # 생성된 rand 에서 temp_list(=각 문장에서 해당하는 주변단어) 빼기, 주변단어를 제외한 랜덤한 단어,,
        rand = list(rand) 
        rand_list.append(rand) # 중심단어 각각의 랜덤단어들의 리스트(6개)를 만들어 주려고 rand_list안에다가 저장한 것
        

# should make neighbor_list length equal to 4. if it is less than 4, add '-1' 
# should make rand_list length equal to 6. if it is less than 6, add '-1' 
neighbor_list = [[-1] * (4 - len(batch)) + batch for batch in neighbor_list] #길이가 비는 곳에  -1 넣어주고 원래 batch를 더해줘서 모양 맞추기 / 맨 마지막
rand_list = [[-1] * (6 - len(batch)) + batch for batch in rand_list] 

# make each list to LongTensor ( Q. not float tensor, why ? )  임베딩 입력값이 인덱스라서 (=정수) 
x_train = torch.LongTensor(center_list) 
y_train = torch.LongTensor(neighbor_list) 
z_train = torch.LongTensor(rand_list) 

# make tensor dataset
dataset = TensorDataset(x_train, y_train, z_train) # 각 단어의 list들을 longtensor로 바꾼것 = train

# use data loader , batchsize = 50, shuffle = True
dataloader = DataLoader(dataset, batch_size= 50, shuffle = True) #  DataLoader , 50 , True
#%%
# for each batch, get loss and learn
for batch_idx, samples in tqdm(enumerate(dataloader)): 
    result = model.forward(samples, True) #주변관계 
    loss = criterion(result, torch.LongTensor([1] * len(result))) #criterion = loss의 합을 구하려고 한건데 보니까 합쳐지지 않음... neighbour일때 라벨 = 1  
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    result = model.forward(samples, False) #랜덤관계
    loss = criterion(result, torch.LongTensor([0] * len(result)))  #random일때 라벨 = 0  
    optimizer.zero_grad()
    loss.backward()
    optimizer.step() 

torch.save(model.state_dict(), './model/model.bin')
#%%
print(torch.sigmoid(torch.tensor([3])))
#%%
criterion = torch.nn.MSELoss()
#%%

criterion = torch.nn.CrossEntropyLoss()
print(criterion(torch.FloatTensor([[0.8,0.2],[0.9,0.1]]),torch.LongTensor([1,1])))
print(criterion(torch.FloatTensor([[0.8,0.2]]),torch.LongTensor([1])))
print(criterion(torch.FloatTensor([[0.9,0.1]]),torch.LongTensor([1])))
#%%



 