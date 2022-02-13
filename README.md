<a href="https://colab.research.google.com/github/majaja068/scClass/blob/main/scClass_demo.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" title="Open in Google Colaboratory">
</a>&nbsp;
<a href="article.md">
  <img src="pic/article_logo.png" height="21.5">
</a>

# scClass：<br>A immune cell classifier tool created by supervised deep learning

Import scRNA_celltype_classifier as:
```
import scClass
```

If you want using our pre-training model, import:
```
from scClass.main import Model
```
that will inhert the torch.nn.Module



### Data Proceesing

#### Attributes
Attributes  |   -
----------  | ----  
data_dir    |the direction where the pre-training model and data are in
out_dir     |the direction where the result are outputed

#### Methods
Methods     |   -
----------  | ----  
read_input(matrix,label)  |read main_matrix and cell_label from path ```matrix``` and  ```label```
write_csv(pred_y,path)    |output classify label```pred_y``` to ```path```
toTorch(data)             |convert ```data``` type <scipy.sparse.csr.csr_matrix> or <numpy.ndarray> to  <torch.Tensor> 
random(matrix)            |kick label<0 and suffle the cell data


### Model Training

#### Attributes

Attributes    | default | -
----------    |----     |--- 
EPOCH         |xx       |HyperParameter 
BATCH_SIZE    |xxxx     |HyperParameter, if your PC out of RAM when training, you can lower down this parameter
LR            |x.xxx    |HyperParameter 
num_in        |xx,xxx   |the number of input  parameter
num_out       |xx       |the number of output parameter
num_acc_train |x,xxx    |since the training set will be huge, the number of data(randomly) calculating training accuracy
ratio_train   |x.x      |the ratio of the data assign to training set 
ratio_val     |x.x      |the ratio of the data assign to validation set, lower this parameter if your PC out of RAM
ratio_test    |x.x      |the ratio of the data assign to testing set, lower this parameter if your PC out of RAM

#### Class

npz_dataloader()      |Imitate pyTorch-DataLoader
----------            | ----  
__init__(matrix,label)|input ```matrix``` and ```label```
__iter__()            |initial step of for loop
__next__()            |recursive step of for loop
train_x()             |randomly (by self.random) get  train set matrix 
train_y()             |randomly (by self.random) get train set label
rand()                |change self.random
val_x                 |attributes of validation set matrix
val_y                 |attributes of validation set label
test_x                |attributes of testing set matrix
test_y                |attributes of testing set label


Model(torch.nn.Module)|pyTorch model
----------            | ----  
__init__()            |inherited from torch.nn.Module
forward(x)            |inherited from torch.nn.Module
save(path)            |saving the model to ```path```
laod(path)            |loading pre-training model on this Model from ```path```

#### Methods

Methods     |   -
----------  | ----  
laod(path)                      |loading pre-training model from ```path```
training(dataset,model)         |training by pyTorch
predict(matrix,model)           |put matrix and model to predict
confusion_matrix(y_pred,y_true) |get a confusion_matrix from ```y_pred``` and ```y_true```
auto_run(load_model,save)       |auto_run demo_training_data.py, ```load_model``` ```save``` are boolean could choosed




