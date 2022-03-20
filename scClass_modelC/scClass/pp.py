import h5py
import scanpy as sc
from scipy import sparse
import pandas as pd
import numpy as np
from pathlib import Path

data_dir = str(Path(__file__).parent)+"/data/"

def read_h5seurat(filename):
    import h5py
    import scanpy as sc
    from scipy import sparse
    import pandas as pd
    ds = h5py.File(filename, 'r')
    m=ds['assays/SCT/counts']
    def get_loop(dir):
        re = {}
        for k in ds[dir].keys():
            if k!='_index':
                re[k] = ds[dir][k][:].astype(str)
        return re 
    def get_once(dir):
        return ds[dir][:].astype(str)
    ad=sc.AnnData(
      X = sparse.csr_matrix((m['data'][:],m['indices'][:],m['indptr'][:])),
      obs = pd.DataFrame(get_loop('meta.data'),index=get_once('meta.data/_index')),
      var = pd.DataFrame(get_once('assays/SCT/features'),columns=['Gene'])
    )
    return ad

def get_type_id(adata,colname_celltype,trans_table,inplace=False):
    Adata = adata if inplace else adata.copy()
    # category matching to speedup
    Adata.obs['origin_cell_type'] = pd.Series(adata.obs[colname_celltype],dtype="category")
    # get transfer_id <0,1,2,...>
    match_seq = [np.where(cell_type==trans_table[:,0])[0][0] for cell_type in Adata.obs['origin_cell_type'].cat.categories]
    match_seq = np.array(match_seq)
    Adata.obs['transfer_id'] = trans_table[:,1][match_seq[Adata.obs['origin_cell_type'].cat.codes]].astype(np.int8)
    # get transfer_cell_type <T cell, B cell, ...>
    classic_table  = pd.read_csv(data_dir+'typeid_classic.csv',header=None).values
    Adata.obs['transfer_cell_type'] = classic_table[:,0][Adata.obs['transfer_id']+1]
    Adata.obs.drop(columns=adata.obs.columns[:Adata.obs.shape[1]-3],inplace=True)
    if not inplace: return Adata.obs[['transfer_id','transfer_cell_type']]
    
def preprocess(adata,filter=True,random=True,normalize=True):
    if filter   : adata = pp_filter(adata)
    if random   : adata = pp_random(adata)
    adata = adata.copy() # csr/csc data rearrangement, or the data got wrong
    if normalize: adata = pp_normalize(adata)
    return adata

def pp_filter(adata,id_colname='modelC id'):  
    print("filter type_id=-1 cell: ",adata.shape[0]," -> ",end="")
    adata = adata[adata.obs[id_colname]>=0,:]
    print(adata.shape[0]," cells")
    return adata

def pp_normalize(adata):
    from scipy import sparse
    print("normalizing data")   
    sc.pp.normalize_total(adata, target_sum=1e5)
    adata.X.data = np.log1p(adata.X.data)
    return adata

def pp_random(adata,seed=None):
    print("shuffling data")
    if seed!=None: np.random.seed(seed)
    rand = np.random.permutation(adata.shape[0])
    adata = adata[rand,:]
    return adata

def transmodel(adata,gene,gene_ref=None,ram=40):
    model_gene = pd.read_csv(data_dir+"ref_gene_v2.csv")
    gene = adata.var[gene].values
    column = 'ensembl_ids'
    if gene_ref in model_gene.keys():
        column = gene_ref
        gene_ref = model_gene[gene_ref].values
    else:
        raise BaseException(f'gene_ref \'{gene_ref}\' is not in model_gene, plese try {list(model_gene.keys())}')
    # new anndata to fit in
    adata2 = sc.AnnData(
        X = sparse.csc_matrix((adata.shape[0],gene_ref.shape[0]),dtype=np.float32),
        var = pd.DataFrame(gene_ref,columns=[column]),
        obs = adata.obs
        )    
    # mapping to model2 <1,5,3,2...>
    print("get gene seq...")
    gene_seq = np.array([-1]*gene.shape[0])
    for i,g in enumerate(gene):
        index = np.nonzero(gene_ref == g)[0]
        if len(index)>0:
            gene_seq[i] = index[0]
        if i%5000==0:
            print(int(i/adata.shape[1]*100),end="% ")
    print(f'\nOf {gene.shape[0]} genes in the input file, {(gene_seq>=0).sum()} were found in the training set of {model_gene.shape[0]} genes.')
    # mapping to model3 rearrange gene index
    print("mapping to model...")    
    a = gene_seq[gene_seq!=-1]
    b = np.arange(gene_seq.shape[0])[gene_seq!=-1]
    batch = np.ceil(38e6*ram/adata.shape[0]).astype(np.int64)
    import warnings; warnings.filterwarnings("ignore")
    for i in range(0,a.shape[0],batch):
        print(int(i/a.shape[0]*100),end="% ")
        adata2[:,a[i:i+batch]].X = adata[:,b[i:i+batch]].X
    return adata2