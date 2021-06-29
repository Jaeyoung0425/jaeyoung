import sentencepiece
import pandas as pd
import csv
#%%
data = pd.read_excel('C:/Users/JalynneHEO/OneDrive - 주식회사 투디지트/바탕 화면/jalyn_excercise.xlsx', engine = 'openpyxl', dtype = object)
data['title'] = data['title'].str.lower()

with open('./news_headline.txt', 'w', encoding='utf8') as f:
    f.write('\n'.join(data['title']))

sentencepiece.SentencePieceTrainer.Train('--input=news_headline.txt '
                               '--model_prefix=headline '
                               '--vocab_size=1000 '
                               '--model_type=bpe '
                               '--max_sentence_length=1000')

#%%
vocab_list = pd.read_csv('headline.vocab', sep='\t', header=None, quoting=csv.QUOTE_NONE)
vocab_list.sample(10)
#%%
print(len(vocab_list))
#%%
lines = [
  "I didn't at all think of it this way.",
  "I have waited a long time for someone to film"
]
for line in lines:
  print(line)
  print(sp.encode_as_pieces(line))
  print(sp.encode_as_ids(line))
  print()
#%%

