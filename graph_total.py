import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
import pandas as pd 
from tqdm import tqdm
import matplotlib.pyplot as plt 
import seaborn as sns
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv
import torch.nn.functional as F
from torch_geometric.data import Data
from collections import defaultdict
import os




class GCNNet(torch.nn.Module):
    def __init__(self):
        super(GCNNet, self).__init__()
        self.conv1 = GCNConv(501, 200)   
        # self.conv2 = GCNConv(400, 30)    
        self.linear1 = nn.Linear(200,20)   
        self.linear2 = nn.Linear(20,3)   
        self.emp = nn.Embedding(58416,4)
        
        self.embcolor = nn.Embedding(643,2)
        self.embsize= nn.Embedding(29,2)
        self.embgroup = nn.Embedding(33,2)
        
        # self.emp.weight.requires_grad = False
        


    def forward(self, data):
        x, edge_index,edge_weight = data.x, data.edge_index,data.edge_attr
        
        
        
        # order = x[:,0]
        pro = x[:, 1:51].long().cuda()
        
        color = x[:, 51:101].long().cuda()
        size = x[:, 101:151].long().cuda()
        group = x[:, 151:201].long().cuda()
        count = x[:, 202].cuda()
        
        order = x[:,0]
        
        index_matrix_pr = torch.arange(200).expand(x.shape[0], -1).cuda()
        index_matrix_csg = torch.arange(100).expand(x.shape[0], -1).cuda()

        
        mask_pr = index_matrix_pr >= (count*4).unsqueeze(1)
        # mask_pr = mask_pr.cuda()
        mask_csg = index_matrix_csg >= (count*2).unsqueeze(1)
        # mask_csg = mask_csg.cuda()

        
        
        pro = pro.reshape(-1)
        color = color.reshape(-1)
        size = size.reshape(-1)
        group = group.reshape(-1)

        
        
        
        emb_pro,emb_color,emb_size,emb_group = self.emp(pro), self.embcolor(color), self.embsize(size), self.embgroup(group)
        emb_pro = emb_pro.view(x.shape[0],-1)
        # emb_pro[mask_pr]=0
        
        emb_color = emb_color.view(x.shape[0],-1)
        # emb_color[mask_csg]=0
        
        emb_size = emb_size.view(x.shape[0],-1)
        # emb_size[mask_csg]=0
        
        emb_group = emb_group.view(x.shape[0],-1)
        # emb_group[mask_csg]=0
         
        count = count.cuda()
        count = count.unsqueeze(1)
        emb_total = torch.concat((emb_pro,emb_color,emb_size,emb_group,count),dim =1 )
        
        edge_index = edge_index.t()
    

        edge_weight =  edge_weight.to(torch.float32).cuda()
        emb_total = emb_total.to(torch.float32).cuda()
        # x = self.conv1(emb_total, edge_index,edge_weight)
        x = self.conv1(emb_total, edge_index)

        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        
        x = self.linear1(x)
        
        
        x = F.relu(x) 
        x = self.linear2(x)


        return order, x
    


feature = pd.read_csv('./data/cus.csv',header=None)



train_data  = pd.read_csv('./data/padding_train.csv')
train_order = pd.read_csv('./data/train_grouped.csv')
train_data.insert(0,'order',train_order['order'])
train_label = train_data['set']

train_data.drop(['set'],axis = 1, inplace= True)



val_data  = pd.read_csv('./data/padding_val.csv')
val_order = pd.read_csv('./data/val_grouped.csv')
val_data.insert(0,'order',val_order['order'])
val_label = val_data['set']
val_data.drop(['set'],axis = 1, inplace= True)


test_data  = pd.read_csv('./data/padding_test.csv')
test_order = pd.read_csv('./data/test_grouped.csv')
test_data.drop(test_data.columns[0], axis=1, inplace=True)
test_data.insert(0,'order',test_order['order'])







def make_graph(adj_path):


    
    value,value1 = adj_path.split('_')[1],adj_path.split('_')[2]
    fet = list(feature[0][int(value):int(value1)])  ## 해당 adj에 포함된 customer 만 추출
    cus_di = {key: i for i, key in enumerate(fet)}
    train = []
    val = []
    test = []
    
    k=0 
   
    total = []
    total_label = []
    order_di = defaultdict(int)
    total_order =[]
    for i in range(train_data.shape[0]):
        
        if train_data['customer'][i] in cus_di.keys():  ## 해당 customer가 포함된 order 행 만 추출 
      
            train.append(k)
            total.append(train_data.iloc[i,:])
          
            order_di[k] = cus_di[train_data['customer'][i]]
            total_label.append(train_label.iloc[i])
            total_order.append(int(train_data.iloc[i,:]['order']))

            k+=1


    for i in range(val_data.shape[0]):  ## 해당 customer가 포함된 order 행 만 추출 
        if val_data['customer'][i] in cus_di.keys():
     
            val.append(k)
            total.append(val_data.iloc[i,:])
           
            order_di[k] = cus_di[val_data['customer'][i]]
            total_label.append(val_label.iloc[i])
            total_order.append(int(val_data.iloc[i,:]['order']))
            k+=1 

    for i in range(test_data.shape[0]):  ## 해당 customer가 포함된 order 행 만 추출 
        if test_data['customer'][i] in cus_di.keys():
        
            test.append(k)
           
            total.append(test_data.iloc[i,:])
            order_di[k] = cus_di[test_data['customer'][i]]
            total_label.append(100)
            total_order.append(int(test_data.iloc[i,:]['order']))
            k+=1

    
    customer_order = defaultdict(list)

    
    for i in range(len(total)):
        f = total[i]
        vals = cus_di[int(f['customer'])]
        customer_order[vals].append(total_order.index(int(f['order'])))        

    
    
    train_mask = torch.zeros(k, dtype=torch.bool)
    val_mask = torch.zeros(k, dtype=torch.bool)
    test_mask = torch.zeros(k, dtype=torch.bool)

  
    train_index = torch.tensor(train, dtype=torch.long)
    val_index = torch.tensor(val, dtype=torch.long)
    test_index = torch.tensor(test, dtype=torch.long)
    
    train_mask[train_index] = True
    val_mask[val_index] = True
    test_mask[test_index] = True

    edge_1 = pd.read_csv(f'./adj/adj_{value}_{value1}_.csv',header=None)
    
    edge_1 = torch.tensor(np.where([edge_1>=0.3], edge_1, 0))
    edge_1 = edge_1.squeeze(0)
    

    
    edge_indices = edge_1.nonzero(as_tuple=False).t().contiguous()
    edge_indices = edge_indices[:, edge_indices[0] < edge_indices[1]]
    edge = edge_indices.t().contiguous()
    
    
    

    edge_indice = []

    for  i in tqdm(range(edge.shape[0])):
        nodes1 = customer_order.get(edge[i][0].item(), [])
        nodes2 = customer_order.get(edge[i][1].item(), [])
        
        ## customer similaruty 값이 0.3보다 높은 order 행 연결 
        for node1 in nodes1:         
            for node2 in nodes2:
                edge_indice.append([node1, node2])


        ## customer가 같은 order 행 연결         
        for i in range(len(nodes1)):
            for j in range(i + 1, len(nodes1)):
                edge_indice.append([nodes1[i], nodes1[j]])

        for i in range(len(nodes2)):
            for j in range(i + 1, len(nodes2)):
                edge_indice.append([nodes2[i], nodes2[j]])
    
    print(len(edge_indice))  
    edge_att = torch.zeros(len(edge_indice))
   
 
    for i in range(len(edge_indice)):
        
        edge_p = edge_indice[i]
        src,dst = edge_p[0],edge_p[1]   

        src = order_di[src]
        dst = order_di[dst] 
        
    
        edge_att[i] = edge_1[src][dst]
    
    
    total = torch.tensor(total)

    data = Data(x=total, edge_index=torch.tensor(edge_indice), edge_attr=edge_att,label = total_label)
    
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    
    print('------------------------------------------------------')
    print(f'train : {len(train)} val : {len(val)} test : {len(test)}')
    return data,value

def train_val_test(data,value):
    
    model = GCNNet().cuda()

    ### to use contextual product embedding 
    model.emp.load_state_dict(torch.load('./model/embproduct_weights.pth'))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)




    best_val_acc = 0

    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        data = data.cuda()
        order, out = model(data)
        
        data.label = torch.tensor(data.label).cuda()
        loss = F.cross_entropy(out[data.train_mask], data.label[data.train_mask])
        loss.backward()
        optimizer.step()

        

        model.eval()
        with torch.no_grad():
            data = data.cuda()
            order, logits = model(data)
            pred = logits[data.val_mask].max(1)[1]
            
            val_acc = pred.eq(data.label[data.val_mask]).sum().item() / data.val_mask.sum().item()
            # print(acc)


        if val_acc > best_val_acc:
            best_val_acc = val_acc
      
            torch.save(model.state_dict(), f'./model/best_model_{value}.pth')

        # print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}, Val Accuracy: {val_acc:.4f}')
    
    print(f'Best Validation Accuracy: {best_val_acc:.4f}')
    
    model = GCNNet().cuda()
    model.load_state_dict(torch.load(f'./model/best_model_{value}.pth'))
    model.eval()
    with torch.no_grad():
        data = data.cuda()
        order, logits = model(data)
        pred = logits[data.test_mask].max(1)[1]
        output = torch.concat((order[data.test_mask].unsqueeze(1),pred.unsqueeze(1)),dim=1)
        output = output.detach().cpu()
        output = output.numpy().astype(int)



        df = pd.DataFrame(output)

        df.to_csv(f'./result/output_{value}.csv',sep='\t',index=False)
        return best_val_acc ,len(data)
        



import glob

# customer adj matrix 를 전부 불러옴 (1000 *  1000) X 342개
path = './adj/*.csv'


csv_files = glob.glob(path)
lists = []


### 342개의 Customer 수에 맞는 그래프 별 node prediction
count = 0
total = 0
total_sum = 0
list_vcc = []
for i in tqdm(range(342)):
    
    da ,va= make_graph(csv_files[i])
    kss,count = train_val_test(da,va)
    total+=count
    total_sum +=kss*count
    value = total_sum/total
    list_vcc.append(kss)
    print(f'val_acc: {value}')

# import csv
# filename = 'val_acc.csv'
# with open(filename, 'w', newline='', encoding='utf-8') as file:
#     writer = csv.writer(file)
    
#     writer.writerows([list_vcc])