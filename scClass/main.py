import numpy as np
import pandas as pd
import scanpy as sc
import loompy as lp
import anndata as ad 
from scipy import sparse 
import matplotlib.pyplot as plt
import torch
import time
import os
import csv
from pathlib import Path

'''
data part - Data Proceesing

Attributes:
 data_dir
 out_dir
 
Methods:
 read_input(matrix,label) 
 write_csv(y_pred,path)    
 toTorch(data)              
 random(matrix)            
'''

# dir
data_dir = str(Path(__file__).parent)+"/data/"
out_dir = "./output/" 

# default: type(matrix)=sparse.npz, type(label)=.csv
# read main_matrix and cell_label from path ```matrix``` and  ```label```
def read_input(matrix,label):
    matrix = sparse.load_npz(matrix)
    label = pd.read_csv(label).values[:,0].astype(np.int8)
    return matrix,label


# output classify label```y_pred``` to ```path```
def save_predict(y_pred,path=None):
    if path is None:
        path = "./output/cell_type_"+time.strftime("%Y%m%d_%H%M%S", time.localtime())+".csv"
    label = ['T-helper cell','cytotoxic T cell','memory B cell','naive B cell','plasma cell',\
             'precursor B cell','pro-B cell','natural killer cell','erythrocyte','megakaryocyte',\
             'monocyte','dendritic cell','bone marrow hematopoietic cell','unknown']   
    label_pred = np.array([label[p] for p in y_pred]) 
    label_all = np.append(y_pred,label_pred).reshape(2,y_pred.shape[0]).T
    print("label stored in: \'",path,"\'",sep="")
    try:  os.stat(out_dir)
    except: os.mkdir(out_dir)    
    with open(path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['index','label'])
        writer.writerows(label_all)
        
        
# convert <scipy.sparse.csr.csr_matrix> or <numpy.ndarray> to <torch.Tensor>
def toTorch(data):
    t = type(data)
    if t==sparse.csr.csr_matrix:
        return torch.tensor(data.toarray())
    elif t==np.ndarray:
        return torch.tensor(data,dtype=torch.long)
    else:
        raise RuntimeError(f'type {t} is not supported on function asTorch(),\n\
                please transfer to <scipy.sparse.csr.csr_matrix> or <numpy.ndarray>')

# kick label<0 and suffle the cell data
def random(matrix,label,seed=None):
    if seed != None:
        np.random.seed(seed)
    print("kick type_id=-1 cell from",matrix.shape[0],"cells ",end="")
    matrix = matrix[label>=0,:]
    label  = label [label>=0]
    print("to",matrix.shape[0],"cells")
    rand = np.random.permutation(matrix.shape[0])
    return matrix[rand,:],label[rand] 
        
        
'''
model part - Model Training

Attributes:
 EPOCH         
 BATCH_SIZE    
 LR            
 num_in        
 num_out       
 num_acc_train 
 ratio_train   
 ratio_val     
 ratio_test    
Classes:
 npz_dataloader()
  
 Model(torch.nn.Module)
  
Methods:
 training(dataset,model)
 predict(matrix,model)
 confusion_matrix(y_pred,y_true)
 auto_run(load_model,save)
'''

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
#         x = self.drop(x)
        x = self.out(x)
        return x
    def save(self,path=None):
        if path is None:
            path = out_dir+"model_"+time.strftime("%Y%m%d_%H%M%S", time.localtime())+".pkl"
        try:  os.stat(out_dir)
        except: os.mkdir(out_dir)
        print("model stored in: \'",path,"\'",sep="")
        torch.save(self, path) 
    def load(self,path = data_dir+"model_train_v2.pkl"):
        m1 = torch.load(path)
        self.load_state_dict(m1.state_dict()) 
        
def load(path = data_dir+"model_train_v2.pkl"):
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
        
# put matrix and model to predict
def predict(matrix,model,threshold=1.8,show=True):
    if type(matrix)==ad._core.anndata.AnnData:
        matrix = torch.tensor(matrix.X.toarray())    
    out = model(matrix)
    y_pred = torch.max(torch.nn.functional.softmax(out, dim=1),dim=1)[1]
    y_pred[out.max(axis=1)[0]<threshold] = -1
    if show:
        plt.hist(y_pred.numpy(),bins=27,range=[-1.2,12.2])
        plt.title("Prediction")
        plt.xlabel("label")
        plt.ylabel("count")
        plt.show()
    return y_pred

threshold=[0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 4.0, 1.0, 1.0, 1.0, 1.0]
def predict_v2(matrix,model):
    threshold_torch = torch.tensor(threshold)
    if type(matrix)==ad._core.anndata.AnnData:
        matrix = torch.tensor(matrix.X.toarray())    
    out = model(matrix)
    y_pred = torch.max(torch.nn.functional.softmax(out, dim=1),dim=1)[1]
    y_pred[out.max(axis=1)[0]<threshold_torch[out.max(axis=1)[1]]] = -1
    return y_pred

def predict_batch(matrix,model,batch_size=None):
    if batch_size is None:
        y_pred = predict_v2(matrix,model)
    else:
        y_pred = torch.empty([0],dtype=torch.int)
        for i in range(0,matrix.shape[0],batch_size):
            print(int(i/matrix.shape[0]*100),end="% ")
            y_pred = torch.cat((y_pred,predict_v2(matrix[i:i+batch_size,:],model)))
    plt.hist(y_pred.numpy(),bins=27,range=[-1.2,12.2])
    plt.title("Prediction")
    plt.xlabel("label")
    plt.ylabel("count")
    plt.show()
    return y_pred

key_label = [
['cd4','reg','helper'],
['cd8','cytotoxic','t cell'],
[['memory','b'],'intermediate'],
[['naive','b'],'b cell'],
[['plasma','~dendritic']],
['pre','pro/pre'],
['pro-B','proB'],
['nk','natural killer'],
['ery'],
['meg','platelet'],
['mono'],
['dc','dendritic'],
['hs','hematopoietic'],
]

index2label = pd.DataFrame(np.array(['unknown','T-helper cell','cytotoxic T cell','memory B cell','naive B cell','plasma cell','precursor B cell','pro-B cell','natural killer cell','erythrocyte','megakaryocyte','monocyte','dendritic cell','BM hematopoietic cell']),index=range(-1,num_out),columns=['label'])


# auto transfer label
def auto_translabel(label_y):
    L = [-1]*label_y.shape[0]
    flag = False
    for argl,label in enumerate(label_y):
        for argk,k in enumerate(key_label):
            for _k in k:        
                if type(_k) == str:
                    if _k in label.lower():
                        L[argl] = argk
                        flag = True
                        break
                else:
                    for __k in _k:
                        if __k[0] == '~':
                            if __k[1:] in label.lower():
                                break
                        else: 
                            if not __k in label.lower():
                                break
                        if __k == _k[-1]:
                            L[argl] = argk
                            flag = True              
            if flag:
                flag = False
                break    
    trans_table = np.empty((label_y.shape[0],2),dtype='O')
    trans_table[:,0] = label_y
    trans_table[:,1] = np.array(L,dtype='str')
    display(pd.DataFrame(trans_table,columns=['label','index'],index=index2label['label'][L]))
    return trans_table
    
# get a confusion_matrix
# def confusion_matrix(y_pred,y_true,_filter=True):
#     from sklearn.metrics import confusion_matrix
#     import matplotlib.ticker as ticker
#     if _filter:
#         y_pred = y_pred[y_true>=0]
#         y_true = y_true[y_true>=0]
#     confmat = confusion_matrix(y_true=y_true, y_pred=y_pred,normalize="true",labels=range(-1,num_out))
#     confmat = np.around(confmat*100, decimals=2)
#     fig, ax = plt.subplots(figsize=(7, 7))
#     ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
#     for i in range(confmat.shape[0]):
#         for j in range(confmat.shape[1]):
#             ax.text(x=j, y=i, s=confmat[i,j], va='center', ha='center')  
    
#     ax.xaxis.set_major_locator(plt.MultipleLocator(1))
#     ax.yaxis.set_major_locator(plt.MultipleLocator(1))  
#     label = ['(-1)unknown','(0)T-helper cell','(1)cytotoxic T cell','(2)memory B cell','(3)naive B cell','(4)plasma cell',\
#              '(5)precursor B cell','(6)pro-B cell','(7)natural killer cell','(8)erythrocyte','(9)megakaryocyte',\
#              '(10)monocyte','(11)dendritic cell','(12)BM hematopoietic cell']
#     plt.xticks(rotation=90)
    
#     ax.xaxis.set_major_locator(ticker.FixedLocator(range(14)))
#     ax.yaxis.set_major_locator(ticker.FixedLocator(range(14)))
#     ax.set_xticklabels(label)
#     ax.set_yticklabels(label)
#     plt.xlabel('predicted label(%)')        
#     plt.ylabel('true label(%)')
#     plt.show() 
    
def confusion_matrix(y_pred,y_true,_filter=True):
    from sklearn.metrics import confusion_matrix
    from matplotlib import ticker
    if _filter:
        y_pred = y_pred[y_true>=0]
        y_true = y_true[y_true>=0]
    confmat = confusion_matrix(y_true=y_true, y_pred=y_pred,normalize="true",labels=range(-1,num_out))
    confmat = np.around(confmat*100, decimals=2)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i, s=confmat[i,j], va='center', ha='center') 
    ax.xaxis.set_major_locator(ticker.FixedLocator(range(num_out+1)))
    ax.yaxis.set_major_locator(ticker.FixedLocator(range(num_out+1)))
    ax.set_xticklabels(range(-1,num_out))
    ax.set_yticklabels(range(-1,num_out))
    plt.title('Confusion Matrix')
    plt.xlabel('predicted label(%)')        
    plt.ylabel('true label(%)')
    plt.close()
    return fig

# display confusion matix
def figure_html(fig):
    from IPython.display import HTML
    import base64
    from io import BytesIO

    tmpfile = BytesIO()
    fig.savefig(tmpfile, format='png')
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
    html_code = f'''
    <div class="flex" style='display: flex;flex-direction: row;'>
      <div>
        <img src='data:image/png;base64,{encoded}' style="width:450px;height:450px;object-fit:none;object-position:-20px -20px;">
      </div> 
      <div style="padding-top:18px;">
        {index2label.to_html()}
      </div>
    '''
    return HTML(html_code)

# another way to viulize label
def heatmap_matrix(y_pred,y_true,label_y,table=None):
    from sklearn.metrics import confusion_matrix
    import matplotlib.ticker as ticker
    confmat = confusion_matrix(y_true=y_true,y_pred=y_pred,normalize="true",labels=range(-1,max(num_out,label_y.shape[0])))
    confmat = np.around(confmat[1:label_y.shape[0]+1,:num_out+1]*100, decimals=2)
    s = np.argsort(confmat.argmax(axis=1))
    confmat = confmat[s,:]
    fig, ax = plt.subplots(figsize=((num_out*0.6), (label_y.shape[0]*0.6)))
    ax.imshow(confmat,cmap=plt.cm.Blues)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]): 
            if table is None:
                ax.text(x=j, y=i, s=confmat[i,j], va='center', ha='center')
            elif  str(j-1)==table[np.where(table[:,0]==label_y[s[i]])[0][0],1]:
                ax.text(x=j, y=i, s=confmat[i,j], va='center', ha='center',bbox=dict(boxstyle='Circle,pad=0.1',fill=False,color='r',lw=2))
            else:
                ax.text(x=j, y=i, s=confmat[i,j], va='center', ha='center')
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1))
    label_x = ['(-1)unknown','(0)T-helper cell','(1)cytotoxic T cell','(2)memory B cell',\
            '(3)naive B cell','(4)plasma cell','(5)precursor B cell','(6)pro-B cell',\
            '(7)natural killer cell','(8)erythrocyte','(9)megakaryocyte','(10)monocyte',\
            '(11)dendritic cell','(12)BM hematopoietic cell']
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    ax.xaxis.set_major_locator(ticker.FixedLocator(range(len(label_x))))
    ax.yaxis.set_major_locator(ticker.FixedLocator(range(label_y.shape[0])))
    ax.set_xticklabels(label_x)
    ax.set_yticklabels(label_y[s])
    plt.title('Heatmap Matrix')
    plt.xlabel('predicted label(%)')        
    plt.ylabel('true label(%)')
    plt.show() 

    
# auto_run
def auto_run(load_model=False,save=True,matrix=None,label=None):
    print('step 1: loading data...')
    matrix,label = read_input(matrix,label)
    print('step 2: processing data... (kick unknown and random)')
    matrix,label = random(matrix,label,seed=0)
    print('step 3: put data in dataset...')
    dataset = npz_dataloader(matrix,label)
    print('step 4: initial training model...')
    model = Model() 
    if load_model:
        print('step 5: load_model=True, loading model...')
        model.load() 
    else:
        print('step 5: load_model=False, training model...')
        training(dataset,model)
    print('step 6: predict the model by testing set...')   
    y_pred = predict_batch(dataset.test_x,model) 
    print('step 7: result for testing set by confusion_matrix...') 
    confusion_matrix(y_pred=y_pred,y_true=dataset.test_y)
    if save:
        print('step 8: saveing model..') 
        model.save()
        
'''
test data preprocessing
'''

def transmodel(adata,gene,gene_ref=None,ram=40):
    model_gene = pd.read_csv(data_dir+"model_gene.csv")
    if gene_ref in model_gene.keys():
        gene_ref = model_gene[gene_ref].values
    else:
        raise BaseException(f'gene_ref \'{gene_ref}\' is not in model_gene, plese try {list(model_gene.keys()[:-1])}')
    adata2 = sc.AnnData(
        X = sparse.csc_matrix((adata.shape[0],gene_ref.shape[0]),dtype=np.float32),
        var = pd.DataFrame(gene_ref,columns=["ensembl_ids"]),
        obs = adata.obs
        )    
    # mapping to model2
    print("get gene seq...")
    gene_seq = np.array([-1]*gene.shape[0])
    for i,g in enumerate(gene):
        index = np.nonzero(gene_ref == g)[0]
        if len(index)>0:
            gene_seq[i] = index[0]
        if i%5000==0:
            print(int(i/adata.shape[1]*100),end="% ")
    print(f'\nOf {gene.shape[0]} genes in the input file, {(gene_seq>=0).sum()} were found in the training set of 45468 genes.')
    print("mapping to model...")
    # mapping to model3
    a = gene_seq[gene_seq!=-1]
    b = np.arange(gene_seq.shape[0])[gene_seq!=-1]
    batch = np.ceil(38e6*ram/adata.shape[0]).astype(np.int64)
    import warnings; warnings.filterwarnings("ignore")
    for i in range(0,a.shape[0],batch):
        print(int(i/a.shape[0]*100),end="% ")
        adata2[:,a[i:i+batch]].X = adata[:,b[i:i+batch]].X
    return adata2

def normalize_simple(adata):
    adata2 = adata.copy()
    adata2.X = adata2.X.tocsr()
    adata_nor = sc.pp.normalize_total(adata2, target_sum=adata2.X.sum(axis=1).mean(),inplace=False)['X']
    sc.pp.log1p(adata_nor)
    adata_nor.data = adata_nor.data/np.log(2).astype(np.float32)
    adata2.X = adata_nor
    return adata2

def get_type_id(label,table):
    type_id = np.zeros(label.shape[0],dtype=np.int8)
    for i,_type in enumerate(label):
        type_id[i] = table[np.where(table[:,0]==_type)[0][0],1]
    return type_id 