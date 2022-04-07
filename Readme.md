# Backup from <a href="https://hub.docker.com/r/hsushihhsueh/scclass">DockerHub</a>
### This image is provided by Institute of Information Science, Academia Sinica, TAIWAN
### Contact information:
Chung-Yen Lin (cylin@iis.sinica.edu.tw); LAB website: http://eln.iis.sinica.edu.tw<br>
Shih-Hsueh Hsu (qqqq471tw@gmail.com)
## Description
#### scClass Dataset
1) BoneMarrow from  Human  Cell  Atlas ([link](https://github.com/HsuShihHsueh/scClass/releases/download/v2022.3.dataset/BoneMarrow.h5ad.gz))<br>
2) CordBlood  from  Human  Cell  Atlas ([link](https://github.com/HsuShihHsueh/scClass/releases/download/v2022.3.dataset/CordBlood.h5ad.gz))<br>
3) PBMC3K from scanpy dataset (origin data from 10x Genomics) ([link](https://github.com/HsuShihHsueh/scClass/releases/download/v2022.3.dataset/PBMC3K.h5ad.gz))<br>
4) PBMC68k from Cell Blast (origin data from 10x Genomics) ([link](https://github.com/HsuShihHsueh/scClass/releases/download/v2022.3.dataset/PBMC68k.h5ad.gz))<br>
5) PBMC_CITE from New  York Genome  Center ([link](https://github.com/HsuShihHsueh/scClass/releases/download/v2022.3.dataset/PBMC_CITE.h5ad.gz))<br>
6) PBMC_scPortal  from Single Cell Portal - Broad Institute ([link](https://github.com/HsuShihHsueh/scClass/releases/download/v2022.3.dataset/PBMC_scPortal.h5ad.gz))<br>
7) Placenta from Cell Blast ([link](https://github.com/HsuShihHsueh/scClass/releases/download/v2022.3.dataset/Placenta.h5ad.gz))<br>
8) HTD_Thymic from Human  Cell  Atlas ([link](https://github.com/HsuShihHsueh/scClass/releases/download/v2022.3.dataset/HTD_Thymic.h5ad.gz))<br>
9) HTD_HSC from Human  Cell  Atlas ([link](https://github.com/HsuShihHsueh/scClass/releases/download/v2022.3.dataset/HTD_HSC.h5ad.gz))<br>
#### Classification Method
1) Model A：machine learing model with single layer MLP<br>
2) Model B：Base on Model A adding batch effect correction by harmony<br>
3) Model C：Base on Model A changing training dataset<br>
4) Model D：subsampling by picking high UMI conuts cells <br>
5) [Seurat](https://www.cell.com/cell/fulltext/S0092-8674(19)30559-8)：Compare with scClass<br>
6) [scCaps](https://www.nature.com/articles/s42256-020-00244-4)：Compare with scClass<br>
## Docker Command
```
# Get scClass Dataset
 Step1. docker pull hsushihhsueh/scclass
 Step2. docker run -d --name=scclass -t -i hsushihhsueh/scclass
 Step3. docker cp scclass:/home/jovyan/scClass/dataset.tar.gz  $(pwd)
 Step4. docker stop scclass
 Step5. tar zxvf dataset.tar.gz 
 //Now scClass dataset files are in local.
# Run scClass notebook
 Step1. docker pull hsushihhsueh/scclass
 Step2. docker run --name=scclass --rm -t -i -p 8888:8888 scclass start-notebook.sh --NotebookApp.token=''
 Step3. jovyan@6e09cc527710:~$ cd scClass; tar zxvf dataset.tar.gz (docker jupyterlab)
//Now scClass can run on JupyterLab
```
## 
**For more detail, please visit our github pages (https://github.com/HsuShihHsueh/scClass).**
