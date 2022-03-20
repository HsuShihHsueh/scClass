import numpy as np
import pandas as pd
import loompy as lp
import seaborn as sns
import csv
from scipy import sparse
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
# python script
from . import loom
# from . import scran

def write_csv(path,array,index=None):
    with open(path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if index!=None:
            writer.writerow(index) 
        writer.writerows(array)

# Step 1: unify to x_CBx_cell_barcode format
def get_u_CellID(ds,df):
    document_id = list(ds.attrs.items())[4][1].split(", ")           # input_id   ex: ec752b0f-b708-4cc7-8557-f36236f93384
    biomaterial_id = list(ds.attrs.items())[6][1].split(", ")        # input_name ex: 1_BM1_cells
    ds_biomaterial_id = np.empty(ds.ca["input_id"].shape, dtype='O') # ds_biomaterial_id, biomaterial_id on ds(loom)
    for x in range(len(document_id)):
        id_indexs = np.where(ds.ca["input_id"] == document_id[x])[0]
        for id_index in id_indexs:
            ds_biomaterial_id[id_index] = biomaterial_id[x]
    u_CellID = ds_biomaterial_id+ds.ca["CellID"]                     # u_CellID = biomaterial_id + CellID
    return u_CellID

# Step 1: mapping csv "annotated_cell_identity.text" label to loom file
def get_label(u_CellID,df,save=None):
    cell_type = df[:,2]                     # annotated_cell_identity.text
    x_CBx_cell_barcode = df[:,1]+df[:,5]    # x_CBx_cell + barcode      
    global func
    def func(u,i):
        if i%10_000==0: print(i,"...",end="\t")
        x = cell_type[x_CBx_cell_barcode == u]
        return "" if x.shape[0]==0 else x[0] 
    with ProcessPoolExecutor(max_workers=8) as executor:
        i = range(u_CellID.shape[0])
        label = executor.map(func,u_CellID,i)
        label2 = np.array(list(label)).astype("O")
    if save!=None:   
        write_csv(path=save,array=label2)
    return label2

# Step 2:
def add_parameter(adata,label,u_CellID):
    mito_genes = adata.var["Gene"].str.startswith('MT')
    percent_mito = adata[:,mito_genes].X.sum(axis=1).A1 / adata.X.sum(axis=1).A1 
    adata.obs["label"] = label                     # "annotated_cell_identity.text" label
    adata.obs["percent_mito"] = percent_mito       # compute fraction of counts in mito genes / all genes
    adata.obs["n_counts"] = adata.X.sum(axis=1).A1 # total counts per cell as observations-annotation
    adata.obs["u_CellID"] = u_CellID               # u_CellID = biomaterial_id + CellID

# Step 2:    
def pick(adata_bm,adata_cb,save_bm=None,save_cb=None):
    adata_bm = adata_bm[adata_bm.obs["label"]!="",:]
    adata_cb = adata_cb[adata_cb.obs["label"]!="",:]
    cell_select = (adata_bm.X.sum(axis=0)!=0) | (adata_cb.X.sum(axis=0)!=0)
    adata_bm = adata_bm[:,cell_select]
    adata_cb = adata_cb[:,cell_select]
    if save_bm!=None:
        adata_bm.write_loom(save_bm+".loom")
        sparse.save_npz(save_bm+".npz",adata_bm.X)
        print("save on :",save_bm,".loom and .npz")
    if save_cb!=None:
        adata_cb.write_loom(save_cb+".loom")
        sparse.save_npz(save_cb+".npz",adata_cb.X)
        print("save on :",save_cb,".loom and .npz")        
    return adata_bm,adata_cb

#Step 3:
def norm_scran(path_loom,path_save):
    scran.package_install()
    matrix = scran.get_matrix(path=path_loom)
    matrix_cluster = scran.get_cluster(matrix,save=path_save+"scran_cluster.csv")
    matrix_sf = scran.get_size_factor(matrix,matrix_cluster,save=path_save+"scran_sf.csv")
    
# Step 4: 
def norm_by_scran_sf(adata,sf,save=None):
    adata_X = adata.X.astype(np.float64)
    deconv_sf = sf.values.flatten()
    deconv_sf = 1/deconv_sf
    sf_i = sparse.csc_matrix((adata.X.shape[0],adata.X.shape[0]),dtype=np.float64)
    sf_i.setdiag(deconv_sf)
    bm3_n1 = sf_i*adata_X
    bm3_n2 = np.log1p(bm3_n1)
    bm3_n2 = bm3_n2/np.log(2)
    bm3_n3 = bm3_n2.astype(dtype=np.float32)
    if save != None:        
        sparse.save_npz(save,bm3_n3)
    return bm3_n3

def norm_by_sf(adata,sf='simple',save=None):
    adata2 = adata.copy()
    adata2.X = adata2.X.tocsc() 
    if sf=='simple':
        sf = adata2.X.sum(axis=1).A1/adata2.X.sum(axis=1).mean()
    adata2.X.data = adata2.X.data/sf[adata2.X.indices]
    adata2.X.data = np.log1p(adata2.X.data)
    adata2.X.data = adata2.X.data/np.log(2).astype(np.float32)
    adata2.X.data = adata2.X.data.astype(np.float32)
    if save != None:        
        sparse.save_npz(save,adata2.X)
    return adata2

# Step 5: 
def get_index(adata,df,save=None):
    label_index = np.zeros(adata.obs["label"].shape[0], dtype=np.int8)
    for x in range(df["identity.text"].shape[0]):
        id_indexs = np.where(adata.obs["label"] == df["identity.text"][x])[0]
        for id_index in id_indexs:
            label_index[id_index] = df["index"][x]
    adata.obs["type_id"] = label_index        
    if save != None:
        adata.write_loom(save)
    return label_index
        
# Step 6: save matrix        
def save_model_data(adata_bm,adata_cb,mix_table,path_save):
    # data_train.npz
    mix_matrix = sparse.vstack([adata_bm.X,adata_cb.X],format="csc")
    sparse.save_npz(path_save+"data_train.npz",mix_matrix)    
    # data_train.csv
    mix_label = np.empty((adata_bm.shape[0]+adata_cb.shape[0],2),dtype='O')
    mix_label[:,0] = np.append(adata_bm.obs["type_id"],adata_cb.obs["type_id"])
    mix_label[:,1] = mix_table["level"][mix_gene[:,0]+1]
    write_csv(path=path_save+"data_train.csv",index=["index","label"],array=mix_label)
    # model_gene.csv
    mix_gene = np.empty((adata_bm.shape[1],2),dtype='O')
    mix_gene[:,0] = [ids[:15] for ids in adata_bm.var["ensembl_ids"]]
    mix_gene[:,1] = adata_bm.var["Gene"]
    write_csv(path=path_save+"model_gene.csv",index=["ensembl_ids","Gene"],array=mix_gene)

