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
from . import scran

# def dataset(path):
#     return "../dataset/"+path

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

# '''
# name:step1.py
# descript: get cell type, and export to label_bm/cb.npy
# '''
# path_bm1_loom = "../dataset/1M-immune-human-immune-10XV2.loom"
# path_cb1_loom = "../dataset/1M-immune-human-blood-10XV2.loom"
# path_bm1_csv = "../dataset/CensusImmune-BoneMarrow-10x_cell_type_2020-03-12.csv"
# path_cb1_csv = "../dataset/CensusImmune-CordBlood-10x_cell_type_2020-03-12.csv"
# # save (as) sparse matrix
# loom.save_matrix_as_npz(path_bm1_loom,dataset("bm1.npz"))
# loom.save_matrix_as_npz(path_cb1_loom,dataset("cb1.npz"))
# # read loom file
# bm1_loom = loom.read(path_bm1_loom,dataset("bm1.npz"))
# cb1_loom = loom.read(path_cb1_loom,dataset("cb1.npz"))
# bm1_lp = lp.connect(path_bm1_loom,mode="r")
# cb1_lp = lp.connect(path_cb1_loom,mode="r")
# # read csv file(anno)
# bm1_csv = pd.read_csv(path_bm1_csv).values
# cb1_csv = pd.read_csv(path_cb1_csv).values
# # unify to x_CBx_cell_barcode format
# bm_u_CellID = get_u_CellID(bm1_lp,bm1_csv)
# cb_u_CellID = get_u_CellID(cb1_lp,cb1_csv)
# # mapping csv "annotated_cell_identity.text" label to loom file
# bm_label =  get_label(bm_u_CellID,bm1_csv,dataset("bm_label.csv"))
# cb_label =  get_label(cb_u_CellID,cb1_csv,dataset("cb_label.csv"))

# '''
# name:step2.py
# descript: add cell type(label), and kick out unlabel cell, then export to bm/cb2.lom+.npz
# '''
# add_parameter(bm1_loom,bm_label,bm_u_CellID)
# add_parameter(cb1_loom,cb_label,cb_u_CellID)
# bm2_loom,cb2_loom = pick(bm1_loom,cb1_loom,save_bm=dataset("bm2"),save_cb=dataset("cb2"))

# '''
# name:step3.py
# descript: normalize by scran
# '''
# norm_scran(path_loom=dataset("bm2.loom"),path_save=dataset("bm"))
# norm_scran(path_loom=dataset("cb2.loom"),path_save=dataset("cb"))

# '''
# name:step4.py
# descript: normalize by scran
# '''
# # data prepare
# bm2_loom = loom.read(dataset("bm2.loom"),dataset("bm2.npz"))
# bm_sf = pd.read_csv(dataset("bm_scran_sf.csv"))
# cb2_loom = loom.read(dataset("cb2.loom"),dataset("cb2.npz"))
# cb_sf = pd.read_csv(dataset("cb_scran_sf.csv"))
# # main
# norm_by_scran_sf(bm2_loom,bm_sf,save=dataset("bm3.npz"))
# norm_by_scran_sf(cb2_loom,cb_sf,save=dataset("cb3.npz"))

# '''
# name:step5.py
# descript: tranfer label to type_id
# '''
# # data prepare
# bm2_loom_norm = loom.read(dataset("bm2.loom"),dataset("bm3.npz"))
# bm_table = pd.read_cs(dataset("label_table/BM.csv"),keep_default_na=False)
# cb2_loom_norm = loom.read(dataset("cb2.loom"),dataset("cb3.npz"))
# cb_table = pd.read_csv(dataset("label_table/CB.csv"),keep_default_na=False)
# # main
# bm_label_index = get_index(bm2_loom_norm,bm_table,save=dataset("bm3.loom"))
# cb_label_index = get_index(cb2_loom_norm,cb_table,save=dataset("cb3.loom"))

# '''
# name:step6.py<br>
# descript: mix BM and CB data and save as data_train.npz(matrix) data_train.csv(label) model_gene.csv(gene)
# '''
# # data prepare
# bm3_loom = loom.read(dataset("bm3.loom"),dataset("bm3.npz"))
# cb3_loom = loom.read(dataset("cb3.loom"),dataset("cb3.npz"))
# mix_table = pd.read_csv(dataset("label_table/type_id.csv"),keep_default_na=False)
# # main
# save_model_data(bm3_loom,cb3_loom,mix_table)
