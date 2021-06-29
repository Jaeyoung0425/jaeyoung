import torch


#%%
# *tensor = 배열의 집합

# *CrossEntropy가 어떻게 구해질까? 
# input이 두개가 들어와야함
# input1 = 각 라벨에 대한 확률 ex) [[0.3,0.7]]
# input2 = 정답라벨 ex) [1]

# input이 여러개가 들어오려면
# input1 = [[0.3,0.7],[0.4,0.6]] -> 라벨이 0일 확률 = 0.3 / 라벨이 1일 확률 = 0.7, 정답은 라벨이 1이니까 라벨 1 확률을 더 높이는 방향으로 학습
# input2 = [1,0]

# crossentropyloss(input1,input2)

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
            output = torch.nn.sigmoid(output)
            output = output.unsqueeze(0)
            output = torch.cat((torch.Tensor([1 - output[0].item()]).unsqueeze(0), output), dim=-1)
        
        except:
            
            return None
        
#%%
import torch
#%%
# output=torch.Tensor([0.7]).unsqueeze(0)

# print(torch.Tensor([1 - output[0].item()]))
# #%%
# output = torch.cat((torch.Tensor([1 - output[0].item()]).unsqueeze(0), output), dim=-1)
# print(output)



