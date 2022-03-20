import numpy as np
import pandas as pd
import scanpy as sc
import loompy as lp
import anndata as ad 
from scipy import sparse 


# speed up loading loom file, it will need a loom file and npz file to store matrix
def read(loom,npz):
    print("reading "+loom+"...")
    ds = lp.connect(loom,mode="r")
    matrix = sparse.load_npz(npz)
    print("tranfer to anndata...")

    # get column
    col = np.zeros(0)
    for x in ds.ca.keys():
        col = np.append(col,ds.ca[x],axis=0)
    col = col.reshape(len(ds.ca.keys()),ds.shape[1])
    col = pd.DataFrame(col.T,columns=ds.ca.keys(),index=[str(i) for i in range(ds.shape[1])])

    # get row
    row = np.zeros(0)
    for x in ds.ra.keys():
        row = np.append(row,ds.ra[x],axis=0)
    row = row.reshape(len(ds.ra.keys()),ds.shape[0])
    row = pd.DataFrame(row.T,columns=ds.ra.keys(),index=[str(i) for i in range(ds.shape[0])])

    adata = ad.AnnData(
        X  = sparse.csc_matrix(([],([],[])), shape=(ds.shape[1],ds.shape[0])),
        obs = col,
        var = row
    )
    adata.X = matrix
    print(adata.X.__getattr__)
    ds.close()
    return adata

def save_matrix_as_npz(path_loom,path_npz,step=5000,dtype=np.uint16):
    # it can change step to balance between calculate_speed and RAM_usage  
    ds = lp.connect(path_loom,mode="r")
    ds_x = sparse.csc_matrix(ds[:,:step].T)
    for start in range(step,ds.shape[1],step):
        print(start,"...",end="\t")
        ds_x = sparse.vstack((ds_x,sparse.csc_matrix(ds[:,start:start+step].T)),format="csc")
    print("\n",ds_x.__len__)
    ds_x = ds_x.astype(dtype=dtype)
    sparse.save_npz(path_npz,ds_x)
    ds.close()