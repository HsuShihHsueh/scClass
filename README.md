# scClass

<a href="https://colab.research.google.com/github/HsuShihHsueh/scClass/blob/v2022.2.modelA/scClass_demo.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" title="Open in Google Colaboratory">
 </a>

#### A immune cell classifier tool created by supervised deep learning

Single-cell RNA sequencing (scRNA-seq) is a novel RNA sequencing
method which can track their RNA-expression in every single cell. However, the
traditional way to annotate cell-type like isolate by flow cytometry or clustering by seurat
is either expensive or inefficient. Here we present scClass, a supervised deep learning
model for annotating celltype on immmune cell. In this article, nine public datasets are
collected used for training set or testing set. Finally, we provide a package for running
scClass on Python and have a demo on [Colab](https://colab.research.google.com/github/majaja068/scRNA-CellType-classifier/blob/main/scClass_demo.ipynb).<br>
whole source code and datasets are available at [Docker Hub](https://hub.docker.com/r/hsushihhsueh/scclass).



## Installation & Import

The package is available using:
```
pip install git+https://github.com/HsuShihHsueh/scClass
```

Import scRNA_celltype_classifier as:
```
import scClass
```

If you want using our pre-training model, import: (which inherit from torch.nn.Module)
```
from scClass.main import Model
```

## Input Requirement

### While Training Model
#### Data Matrix File
To save the RAM, we use```scipy.sparse``` of shape (n_genes, n_cells) to store data matrix file. For example:
   /  | 1_cell | 2_cell | 3_cell |...
------|--------|--------|--------|----
1_gene|       0|       0|       1|...
2_gene|       2|       0|       0|...
3_gene|       0|       0|       0|...
...   |       .|       .|       .|...

#### Cell Type File
We use 0 ~ 12 to represent 13 cell types, which are:

index |cell_type
------|-------------------------------
-1	  |unknown
0	  |T-helper cell
1	  |cytotoxic T cell
2	  |memory B cell
3	  |naive B cell
4	  |plasma cell
5	  |precursor B cell
6	  |pro-B cell
7	  |natural killer cell
8	  |erythrocyte
9	  |megakaryocyte
10	  |monocyte
11	  |dendritic cell
12	  |bone marrow hematopoietic cell 

save the cell types list as```numpy.array```.
#### dataloader
Once the data matrix file and cell type file are prepared, use the function below to load in:
```
dataset = scClass.npz_dataloader(matrix,label)
```

### While Doing Classification by Using Pre-training Model
Put ```anndata.AnnData``` data :
```
y_pred = scClass.predict_batch(adata,model,batch_size=8_000)
```
Save Prediction:
```
scClass.save_predict(y_pred,path)
```
the prediction on path ```./output/cell_type_xxxxxxxx_xxxxxx.csv``` will look like:

```
index,label
0,T-helper cell
2,memory B cell
0,T-helper cell
10,monocyte
7,natural killer cell
0,T-helper cell
1,cytotoxic T cell
1,cytotoxic T cell
1,cytotoxic T cell
...
```

## Data Proceesing

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
EPOCH         |10       |HyperParameter 
BATCH_SIZE    |5000     |HyperParameter, if your PC out of RAM when training, you can lower down this parameter
LR            |0.001    |HyperParameter 
optimizer     |Adam     |
num_in        |45,468   |the number of input  parameter
num_out       |13       |the number of output parameter
num_acc_train |1,000    |since the training set will be huge, the number of data(randomly) calculating training accuracy
ratio_train   |0.7      |the ratio of the data assign to training set 
ratio_val     |0.2      |the ratio of the data assign to validation set, lower this parameter if your PC out of RAM
ratio_test    |0.1      |the ratio of the data assign to testing set, lower this parameter if your PC out of RAM

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




