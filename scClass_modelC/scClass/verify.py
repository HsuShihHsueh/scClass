import matplotlib.pyplot as plt
import torch
import anndata as ad 
import scanpy as sc
import numpy as np
import pandas as pd


threshold = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0]
def predict_batch(matrix,model):
    threshold_torch = torch.tensor(threshold)
    if type(matrix)==ad._core.anndata.AnnData:
        matrix = torch.tensor(matrix.X.toarray())    
    out = model(matrix)
    y_pred = torch.max(torch.nn.functional.softmax(out, dim=1),dim=1)[1]
    out = (out-out.mean(1,keepdim=True))/out.std(1,keepdim=True)
    y_pred[out.max(axis=1)[0]<threshold_torch[out.max(axis=1)[1]]] = -1
    return y_pred

def predict(matrix,model,batch_size=None):
    print(threshold)
    if batch_size is None:
        y_pred = predict_batch(matrix,model)
    else:
        y_pred = torch.empty([0],dtype=torch.int)
        for i in range(0,matrix.shape[0],batch_size):
            print(int(i/matrix.shape[0]*100),end="% ")
            y_pred = torch.cat((y_pred,predict_batch(matrix[i:i+batch_size,:],model)))
    plt.hist(y_pred.numpy(),bins=23,range=[-1.2,10.2])
    plt.title("Prediction")
    plt.xlabel("label")
    plt.ylabel("count")
    plt.show()
    return y_pred

def save_predict(filename,y_pred):
    print('saved at: '+filename)
    np.savetxt(filename,y_pred.numpy(),fmt='%i')

def confusion_matrix(y_pred,y_true,filter=True):
    from sklearn.metrics import confusion_matrix
    from matplotlib import ticker
    if filter:
        y_pred = y_pred[y_true>=0]
        y_true = y_true[y_true>=0]
    labels_arange = range(-1,max(y_pred.max(),10)+1)
    confmat = confusion_matrix(y_true=y_true, y_pred=y_pred,normalize="true",labels=labels_arange)
    confmat = np.around(confmat*100, decimals=2)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i, s=confmat[i,j], va='center', ha='center') 
    ax.xaxis.set_major_locator(ticker.FixedLocator(range(len(labels_arange))))
    ax.yaxis.set_major_locator(ticker.FixedLocator(range(len(labels_arange))))
    ax.set_xticklabels(labels_arange)
    ax.set_yticklabels(labels_arange)
    plt.title('Confusion Matrix')
    plt.xlabel('predicted label(%)')        
    plt.ylabel('true label(%)')
    plt.close()
    print('total acc:',(np.array(y_pred)==np.array(y_true)).sum()/y_true.shape[0]*100,'%')
    return fig

def figure_html(fig):
    from IPython.display import HTML
    import base64
    from io import BytesIO
    index2label = pd.DataFrame(np.array(['unknown','T-helper cell','cytotoxic T cell','memory B cell','naive B cell','plasma cell','natural killer cell','erythrocyte','megakaryocyte','monocyte','dendritic cell','BM hematopoietic cell']),index=range(-1,11),columns=['label'])
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

def plot_performance(y_pred,y_true):
    from sklearn.metrics import precision_score,recall_score,f1_score
    Y_pred = np.array(y_pred[y_true>=-1])
    Y_true = y_true[y_true>=-1]
    accuary = (Y_pred==Y_true).sum()/Y_pred.shape[0]
    precision = precision_score(Y_pred,Y_true,average=None)
    precision = precision[precision>0].mean()
    recall = recall_score(Y_pred,Y_true,average=None)
    recall = recall[recall>0].mean()
    f1 = f1_score(Y_pred,Y_true,average=None)
    f1 = f1[f1>0].mean()
    table = pd.DataFrame([
      ['accuary',accuary],
      ['precision',precision],
      ['recall',recall],
      ['f1-score',f1]
    ])
    table.columns = ['Performance','Value']
    display(table) 