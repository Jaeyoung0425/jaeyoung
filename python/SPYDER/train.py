import torch
import pandas as pd
from model import CBOW
from tqdm import tqdm
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
vocab_file = './headline.model'
sp.Load(vocab_file)

data = pd.read_excel('./data/jalyn_excercise.xlsx', engine = 'openpyxl', dtype = object)
data['title'] = data['title'].str.lower()
title_list = data['title'].tolist()
title_list = [sp.EncodeAsIds(i) for i in title_list]


model = CBOW(sp.GetPieceSize(), 50, 5)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)


for sentence in tqdm(title_list):
    for word_num in range(len(sentence)):
        output = model.forward(sentence, word_num)
        if output is None:
            pass
        else:
            output = output.unsqueeze(0)
            loss = criterion(output, torch.LongTensor([sentence[word_num]]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

torch.save(model.state_dict(), './model/model3.bin')


