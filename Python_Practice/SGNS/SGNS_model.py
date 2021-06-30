import torch
import time
#%%
#Word to Vec With Negative Sampling
class SGNS(torch.nn.Module):
    def __init__(self, D_dim, W_dim):
        super(SGNS, self).__init__()
        self.dict_len = D_dim 
        #first embedding layer, for center word
        self.embedding1 = torch.nn.Embedding(D_dim, W_dim) #단어 개수, 임베딩 시킬 벡터의 차원
        #second embedding layer, for neighbor word
        self.embedding2 = torch.nn.Embedding(D_dim, W_dim) 

    # number_1 is the number of center vocab, number_2 is the number of neighbor vocab
    # IF desc is True : it means use neighbor-word relation 
    # IF desc is False : it means use random-word relation
    def forward(self, dataset, desc):
        
        try:
            x_temp, y_temp, z_temp = dataset #center, neighbour, rand
            if desc is True: 
                result = torch.empty(0)  
                for i in range(len(x_temp)): #
                    for j in range(len(y_temp[i])): #중심단어 i에 해당하는 neighbour words가 4개니까 j로 접근하기
                        if y_temp[i][j].item() == -1: 
                            pass
                        else:
                            #if we use embedding layer, if you just put the number in dictionary, it returns the embedding vector
                            projection = self.embedding1(torch.tensor([x_temp[i].item()])) #.item = i번째 해당하는 숫자만 꺼내기
                            #.item 한 후 torch.tensor을 굳이 한 이유는 그냥 x_temp[i] 했을 땐 안돌아서,,,
                            projection2 = self.embedding2(torch.tensor([y_temp[i][j].item()])) # y_temp[j].item()

                            #inner product two projection vector 
                            output = torch.dot(projection.squeeze(0), projection2.squeeze(0)) 

                            #use sigmoid function
                            output = torch.sigmoid(output) 
                            output = output.unsqueeze(0) 
                            output = torch.cat((torch.Tensor([1 - output[0].item()]), output), dim=-1)
                            output = output.unsqueeze(0)
                            result = torch.cat((result, output)) #result = torch.empty(0)
            
            if desc is False: 
                result = torch.empty(0) 
                for i in range(len(x_temp)):
                    for j in range(len(z_temp[i])):
                        if z_temp[i][j].item() == -1: # -1이 나오는 이유는 train에서 모양 맞춰주려고 -1해준걸 여기에서 pass시키기 
                            pass
                        else:
                            projection = self.embedding1(torch.tensor([x_temp[i].item()]))
                            projection2 = self.embedding2(torch.tensor([z_temp[i][j].item()]))
                            
                            output = torch.dot(projection.squeeze(0), projection2.squeeze(0))
                            output = torch.sigmoid(output)
                            output = output.unsqueeze(0)
                            output = torch.cat((torch.Tensor([1 - output[0].item()]), output), dim=-1)
                            output = output.unsqueeze(0)
                            result = torch.cat((result, output))

            return result

        except Exception as e:
           
            print(e)
            
            return None 



#%% 
class SGNS(torch.nn.Module):
    def __init__(self, D_dim, W_dim): 
        super(SGNS, self).__init__()
        self.dict_len = D_dim
        self.linear1 = torch.nn.Linear(D_dim, W_dim)
        self.linear2 = torch.nn.Linear(D_dim, W_dim)
        
    def forward(self, number_1, number_2): #number_1 = 문장에서 중심단어의 위치
        try:
            temp_tensor = torch.Tensor([0]*self.dict_len) 
            temp_tensor[number_1] = 1
            
            projection = self.linear1(temp_tensor)
            
            temp_tensor2 = torch.Tensor([0]*self.dict_len)
            temp_tensor2[number_2] = 1 
            
            projection2 = self.linear2(temp_tensor2)
            
            output = torch.dot(projection, projection2)
            output = torch.sigmoid(output)
            output = output.unsqueeze(0)
            output = torch.cat((torch.Tensor([1 - output[0].item()]).unsqueeze(0), output), dim=-1)
        
        except:
            return None
