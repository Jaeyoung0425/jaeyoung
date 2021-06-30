import torch
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
#%%
data = pd.read_excel('C:/Users/JalynneHEO/OneDrive - 주식회사 투디지트/바탕 화면/jalyn_excercise.xlsx', engine = 'openpyxl', dtype = object)
#%%
title_list = data['title'].tolist() # data의 title column을 list로 만들기
title_list = [i.split() for i in title_list] # 문장들을 단어로 split하기
rm_list = ['(',')',',','.'] 
for _ in range(len(title_list)): # 문장을 단어로 쪼갠것들의 개수를 범위로 설정하기 / 각 단어 = _
    title_list[_] = [i.lower() for i in title_list[_]] #단어들을 소문자로 설정해서 list 업데이트 하기
    for rm in rm_list:
        title_list[_] = [i.replace(rm,'') for i in title_list[_]] #rm들을 공백으로 바꿔서 list 업데이트 하기 

dict_list = [] 
for _ in title_list: #title_list안에 있는 쪼개진 문장들을 _라고 설정하기
    dict_list = dict_list + _ # title_list에 들어있는 쪼개진 단어들의 list들을 dict_list에 넣어서 단어사전 만들기
    # 
dict = list(set(dict_list)) # set = 중복되지 않은 원소를 구할 때 사용함, 중복되지 않게 dictionary만들어주기

model = CBOW(len(dict), 50, 2)  

#criterian = CrossEntropyLoss
criterion = torch.nn.CrossEntropyLoss() # nn.CrossEntropyLoss()

#optimizer = Adam, learning rate = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=0.01) # optim.Adam(model.parameters(), lr=0.01)

for sentence in tqdm(title_list): 
    for word_num in range(len(sentence)): #쪼개진 단어의 개수를 범위로 하기, sentence =  쪼개진 단어들의 list
        sentence_encode = [dict.index(i) for i in sentence] # 단어들을 인코딩 해주기 = 숫자로 바꿔서 컴퓨터가 이해할 수 있게 하기

        output = model.forward(sentence_encode, word_num) # 
        if output is None:  #none이 뜨는 경우가 발생해서 if 로 처리해서 pass시킴
            pass
        else:
            output = output.unsqueeze(0) # squeeze와 반대로 1인 차원을 생성하는 함수, 어느 차원에 1인 차원을 생성할 지 지정해야 함, [[]]이런 형태여야함
            #[[ ]] = CrossEntropy 구하는 공식
            loss = criterion(output, torch.LongTensor([sentence_encode[word_num]])) # output=예측값, torch.LongTensor()=실제값
            
            optimizer.zero.grad() #zero.grad()
            loss.backward() #backward()
            optimizer.step() #step()

torch.save(model.state_dict(), 'C:/Users/JalynneHEO/OneDrive - 주식회사 투디지트/바탕 화면/CBOW')

