import torch
import pandas as pd
from model import CBOW
from tqdm import tqdm

data = pd.read_excel('./data/jalyn_excercise.xlsx', engine = 'openpyxl', dtype = object)
title_list = data['title'].tolist()
title_list = [i.split() for i in title_list]
rm_list = ['(',')',',','.']
for _ in range(len(title_list)):
    title_list[_] = [i.lower() for i in title_list[_]]
    for rm in rm_list:
        title_list[_] = [i.replace(rm,'') for i in title_list[_]]

dict_list = []
for _ in title_list:
    dict_list = dict_list + _
dict = list(set(dict_list))


model = CBOW(len(dict), 50, 2)

#criterian = CrossEntropyLoss
criterion = '???'

#optimizer = Adam, learning rate = 0.01
optimizer = '???'


for sentence in tqdm(title_list):
    for word_num in range(len(sentence)):
        for _ in sentence:
            sentence_encode = [dict.index(i) for i in sentence]

        output = model.forward('???', '???')
        if output is None:
            pass
        else:
            output = output.unsqueeze(0)
            loss = criterion('???', torch.LongTensor([sentence_encode[word_num]]))

            optimizer.'???'
            loss.'???'
            optimizer.'???'

torch.save(model.state_dict(), './model/model.bin')

