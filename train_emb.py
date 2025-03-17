import pandas as pd
from collections import defaultdict
import itertools
import torch
from torch_geometric.utils import *
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt 
import seaborn as sns
from collections import defaultdict
import wandb
import random
from sklearn.model_selection import train_test_split



import wandb
import random



class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx].float(), self.labels[idx]



train_set = pd.read_csv('./data/train_set.csv')
dicts = defaultdict(list)
for i in range(1866383):
    dicts[train_set.iloc[i,0]].append(train_set.iloc[i,1])
pos = []
k = 0 
for key,value in dicts.items():
    klist = list(itertools.combinations(value,2))
    for i in range(len(klist)):
        pos.append(klist[i])
pos = set(pos)
sorted_pos = [sorted(sub) for sub in pos]
pos_train = torch.tensor(sorted_pos)
neg_train = negative_sampling(pos_train.t().contiguous(),58415,len(pos_train))

neg_train = neg_train.t().contiguous()
pos_label = torch.ones(pos_train.shape[0])
neg_label = torch.zeros(pos_train.shape[0])

data = torch.cat([pos_train,neg_train],dim = 0)
label = torch.cat([pos_label,neg_label],dim = 0)



X_train, X_val, y_train, y_val = train_test_split(data,label,test_size=0.2,shuffle=True)




dataset = CustomDataset(X_train, y_train)
dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)

val_dataset = CustomDataset(X_val, y_val)
val_dataloader = DataLoader(val_dataset, batch_size=1024 , shuffle=True)




class MLP(nn.Module):
    def __init__(self, emb_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        # self.emborder = nn.Embedding(849190,emb_size[5])
        self.embproduct = nn.Embedding(58416,emb_size[0])
        # self.embcustomer = nn.Embedding(342040,emb_size[1])

      
        
        self.layers = nn.Sequential(
            
            nn.Linear(4, hidden_size[0]),
            # nn.BatchNorm1d(15),
            nn.ReLU(),
            # nn.Dropout(p=0.3),
            nn.Linear(hidden_size[0], hidden_size[1]),
            # nn.BatchNorm1d(10),
            nn.ReLU(),
            # nn.Dropout(p=0.3),
            nn.Linear(hidden_size[1], num_classes),
            # nn.Softmax(dim=1)
        )
        self.linear_1 = nn.Linear(13,5)
        self.linear_2 = nn.Linear(7,5)
    
    def forward(self, x):
        
        p1 = x[:,0].long().cuda()
        p2 = x[:,1].long().cuda()
        p1 = self.embproduct(p1)
        p2 = self.embproduct(p2)
        x = p1 * p2
        
        x = self.layers(x)
        
        return x
        
    
    
    
model = MLP(emb_size=[4], hidden_size=[5,3], num_classes=2)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

model = model.cuda()

num_epochs = 5

for epoch in tqdm(range(num_epochs)):
    model.train()
    conf_matrix = torch.zeros(2, 2, dtype=torch.int64)
    train_loss = 0
    for inputs, targets in tqdm(dataloader):
    
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
    
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(dataloader)
    # wandb.log({"train_loss": train_loss})
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    model.eval()
    

    with torch.no_grad():
        correct = 0
        total = 0
        total_len = 0
        val_loss = 0
        for inputs, targets in val_dataloader:
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs= model(inputs)
            loss = criterion(outputs, targets)
            # count = count.squeeze(1)
            _, predicted = torch.max(outputs.data, 1)
            # new_predicted = torch.zeros((predicted.shape))
        
            # for i in range(int(inputs.shape[0])):
            #     dictionary[int(order[i])].append(predicted[i])
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            val_loss += loss.item()
            # for t, p in zip(targets.view(-1), predicted.view(-1)):
            #     conf_matrix[t.long(), p.long()] += 1
        val_loss /= len(val_dataloader)
        
                
    print(f'Accuracy: {100 * correct / total}%')
    


# i = 0
# for inputs, targets in tqdm(dataloader):

#     inputs = inputs
#     i +=1 
#     if i == 1:
#         break 
# x = inputs.cuda()

# torch.onnx.export(model,x,'model_emb.onnx',input_names=['p1 & p2'],ou5tput_names =['yhat'])
torch.save(model.embproduct.state_dict(), './model/embproduct_weights.pth')