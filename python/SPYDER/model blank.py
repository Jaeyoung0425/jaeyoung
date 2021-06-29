# Using CBOW, train W2V
import torch


class CBOW(torch.nn.Module):
    def __init__(self, D_dim, W_dim, W_len): #W_len = Window_size, D_dim = 단어 개수, W_dim = 차원  
        super(CBOW, self).__init__()
        self.dict_len = D_dim  #
        self.window_screen = W_len # window_size 설정
        self.linear1 = torch.nn.Linear(D_dim, W_dim) #첫번째 W
        self.linear2 = torch.nn.Linear(W_dim, D_dim) #두번째 W 

    def forward(self, sentence, number):
        try:
            if number == 0:
                target_sentence = sentence[number: number + 3] 
            elif number == 1:
                target_sentence = sentence[number - 1 : number + 3] #number-1 : number+3
            else:
                target_sentence = sentence[number - 2: number + 3]
            
            hidden_layer = 0 # 단어 네개가 모여서 m으로 오면 hidden layer가 됨
            for _ in range(len(target_sentence)): #target_sentence  
                if sentence.index(target_sentence[_]) == number : # number #주변단어만 학습해야하니까 중심단어는 pass 
                    pass 
                else:
                    temp_tensor = torch.Tensor([0] * self.dict_len) # 단어들의 벡터를 만들어야 하니까 총 단어길이만큼 tensor를 0으로 만들기
                    temp_tensor[target_sentence[_]] = 1 
                    hidden_layer += (self.linear1(temp_tensor)) # 단어하나씩 hidden_layer에 더해주기

            hidden_layer = hidden_layer / 4 # v=(v_1+v_2+v_3+v_4)/2*window_size 

            #activation
            hidden_layer = torch.nn.tanh(hidden_layer) #torch.nn.tanh

            output_layer = self.linear2(hidden_layer) 
            return output_layer
        
        except:
            return None

#%%
sentence = ['AAL', 'Is', 'Lagging' ,'the', 'Benchmarks', 'But' ,'That', 'a', 'Reason', 'to', 'Own', 'It']
#%%
print(sentence.index('a'))
#%%
print(sentence[0])
#%%

