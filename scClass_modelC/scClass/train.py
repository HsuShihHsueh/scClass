import torch
import time
import scanpy as sc
import pandas as pd
import os
import numpy as np
import anndata as ad 
from scipy import sparse 
from pathlib import Path


# dir
data_dir = str(Path(__file__).parent)+"/data/"
out_dir = "./output/" 

# hyper parameter
EPOCH = 10    
BATCH_SIZE = 5000
LR = 0.001
num_in,num_out = 45_468, 13
ratio_train,ratio_val,ratio_test = 0.7,0.2,0.1

# dataset
class npz_dataloader():
    def __init__(self,matrix,label):
        print("loading data, it will take time")
        # data spilt
        global num_train,num_val,num_test,num_in,num_out,num_acc_train
        num_train = int(matrix.shape[0]*ratio_train)
        num_val = int(matrix.shape[0]*ratio_val)
        num_test = int(matrix.shape[0]*ratio_test)
        num_in = matrix.shape[1]
        num_out = label.max()+1 
        num_acc_train = 1000  
        # dataset
        self.train_matrix = matrix[:num_train,:]
        self.train_label = label[:num_train]
        self.val_x = torch.tensor(matrix[num_train:num_train+num_val,:].toarray())
        self.val_y = torch.tensor(label[num_train:num_train+num_val],dtype=torch.long)
        self.test_x = torch.tensor(matrix[num_train+num_val:num_train+num_val+num_test,:].toarray())
        self.test_y = torch.tensor(label[num_train+num_val:num_train+num_val+num_test],dtype=torch.long)
        # parameter
        self.length = self.train_matrix.shape[0]
        self.batch_size = BATCH_SIZE
    def train_x(self):
        return torch.tensor(self.train_matrix[self.random,:].toarray(),dtype=torch.float)
    def train_y(self):
        return torch.tensor(self.train_label[self.random],dtype=torch.long)
    def rand(self):
        self.random = np.random.randint(num_train, size=(num_acc_train))
    def __iter__(self):
        self.index = 0 
        return self
    def __next__(self):
        self.index += self.batch_size
        if self.index < self.length+self.batch_size:
            return (
                torch.squeeze(torch.tensor(self.train_matrix[self.index-self.batch_size:self.index,:].toarray(),dtype=torch.float)),
                torch.tensor(self.train_label[self.index-self.batch_size:self.index],dtype=torch.long)
            )  
        else:
            raise StopIteration
    def __len__(self):
        return (self.length // self.batch_size)+1
    
# model
class Model(torch.nn.Module):
    def __init__(self,seed=1):
        super(Model, self).__init__()
        torch.manual_seed(seed)
        self.out = torch.nn.Linear(num_in,num_out)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LR) 
        self.drop = torch.nn.Dropout(p=0.4)
        self.loss_func = torch.nn.CrossEntropyLoss()
    def forward(self,x):       
        x = self.out(x)
        return x
    def save(self,path=None):
        if path is None:
            path = out_dir+"model_"+time.strftime("%Y%m%d_%H%M%S", time.localtime())+".pkl"
        try:  os.stat(out_dir)
        except: os.mkdir(out_dir)
        print("model stored in: \'",path,"\'",sep="")
        torch.save(self, path) 
    def load(self,path = data_dir+"model_default.pkl"):
        m1 = torch.load(path)
        self.load_state_dict(m1.state_dict()) 
        
def load(path = data_dir+"model_default.pkl"):
    print("loading model from: ",path)
    return torch.load(path)

# training by pyTorch
def training(dataset,model):
    for e in range(EPOCH):
      for step,(b_x,b_y) in enumerate(dataset):
        out = model(b_x)
        loss = model.loss_func(out, b_y) 

        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()
        del  b_x,b_y,out,loss
        if step%10==0:
            print(step,end=" ")
      print()
      if e % 1 ==0:
        # train
        dataset.rand()
        out = model(dataset.train_x())
        y_pred = torch.max(torch.nn.functional.softmax(out, dim=1),dim=1)[1]
        accuracy = (y_pred == dataset.train_y()).sum().item() / num_acc_train
        loss = model.loss_func(out, dataset.train_y())
        print("epoch",e,"\t| loss:%.8f"%loss.item(),"| training accuracy:%.8f"%accuracy,end="")
        # test
        out_val = model.eval()(dataset.val_x)
        y_pred_val = torch.max(torch.nn.functional.softmax(out_val, dim=1),dim=1)[1]
        accuracy = (y_pred_val == dataset.val_y).sum().item() / num_val
        loss = model.loss_func(out_val, dataset.val_y)
        print("| loss:%.8f"%loss.item(),"| validation accuracy:%.8f"%accuracy)
        
