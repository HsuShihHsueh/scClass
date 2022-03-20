#! -*- coding: utf-8 -*-
# the capsule parts refer to https://github.com/bojone/Capsule and https://kexue.fm/archives/5112

from Visualization_Capsule_Keras import *
from keras import utils
from keras.models import Model
from keras.layers import *
from keras import backend as K
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame
from sklearn.model_selection import train_test_split

randoms = 30
cell_type = ['B cells','CD14+ Monocytes','CD4+ T cells','CD8+ T cells','Dendritic Cells',
 'FCGR3A+ Monocytes','Megakaryocytes','Natural killer cells']

data = np.load('data/PBMC_data.npy')
labels = np.load('data/PBMC_celltype.npy')

num_classes = 8

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size = 0.1, random_state= randoms)
Y_test = y_test
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)

z_dim = 16

input_size = x_train.shape[1]
x_in = Input(shape=(input_size,))
x = x_in
x1 = Dense(z_dim, activation='relu')(x_in)
x2 = Dense(z_dim, activation='relu')(x_in)
x3 = Dense(z_dim, activation='relu')(x_in)
x4 = Dense(z_dim, activation='relu')(x_in)
x5 = Dense(z_dim, activation='relu')(x_in)
x6 = Dense(z_dim, activation='relu')(x_in)
x7 = Dense(z_dim, activation='relu')(x_in)
x8 = Dense(z_dim, activation='relu')(x_in)
x9 = Dense(z_dim, activation='relu')(x_in)
x10 = Dense(z_dim, activation='relu')(x_in)
x11 = Dense(z_dim, activation='relu')(x_in)
x12 = Dense(z_dim, activation='relu')(x_in)
x13 = Dense(z_dim, activation='relu')(x_in)
x14 = Dense(z_dim, activation='relu')(x_in)
x15 = Dense(z_dim, activation='relu')(x_in)
x16 = Dense(z_dim, activation='relu')(x_in)

encoder1 = Model(x_in, x1)
encoder2 = Model(x_in, x2)
encoder3 = Model(x_in, x3)
encoder4 = Model(x_in, x4)
encoder5 = Model(x_in, x5)
encoder6 = Model(x_in, x6)
encoder7 = Model(x_in, x7)
encoder8 = Model(x_in, x8)
encoder9 = Model(x_in, x9)
encoder10 = Model(x_in, x10)
encoder11 = Model(x_in, x11)
encoder12 = Model(x_in, x12)
encoder13 = Model(x_in, x13)
encoder14 = Model(x_in, x14)
encoder15 = Model(x_in, x15)
encoder16 = Model(x_in, x16)

x = Concatenate()([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16])
encoder = Model(x_in,x)
x = Reshape((16, z_dim))(x)
capsule = Capsule(num_classes, 16, 3, False)(x)
output = capsule

model = Model(inputs=x_in, outputs=output)
model.compile(loss=lambda y_true,y_pred: y_true*K.relu(0.9-y_pred)**2 + 0.25*(1-y_true)*K.relu(y_pred-0.1)**2,
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

model.load_weights('data/Demo_PBMC.weights')

encoder1weight = encoder1.get_weights()
encoder2weight = encoder2.get_weights()
encoder4weight = encoder4.get_weights()
encoder5weight = encoder5.get_weights()
encoder6weight = encoder6.get_weights()
encoder8weight = encoder8.get_weights()
encoder9weight = encoder9.get_weights()
encoder10weight = encoder10.get_weights()
encoder11weight = encoder11.get_weights()
encoder12weight = encoder12.get_weights()
encoder14weight = encoder14.get_weights()
encoder15weight = encoder15.get_weights()
encoder16weight = encoder16.get_weights()

encoder1weight0 = encoder1weight[0]
encoder2weight0 = encoder2weight[0]
encoder4weight0 = encoder4weight[0]
encoder5weight0 = encoder5weight[0]
encoder6weight0 = encoder6weight[0]
encoder8weight0 = encoder8weight[0]
encoder9weight0 = encoder9weight[0]
encoder10weight0 = encoder10weight[0]
encoder11weight0 = encoder11weight[0]
encoder12weight0 = encoder12weight[0]
encoder14weight0 = encoder14weight[0]
encoder15weight0 = encoder15weight[0]
encoder16weight0 = encoder16weight[0]

'''
#Fig 2 DF
#######################################################################################################
#Masking selected cell type specific genes
selectweight =encoder4weight0
pca = PCA(n_components=16)
pca.fit(selectweight)
weightpca = pca.transform(selectweight)

removed = []
for i in range(3346):
    if weightpca[i, 0] >0.0845:
        removed.append(i)
        rownum = x_test.shape[0]
        x_test[:, i] = np.zeros(rownum)
'''

#FIG1 CD
###################################################################################################
#type averaged coupling coefficients
Y_pred = model.predict(x_test)

coupling_coefficients_value = {}
count = {}
for i in range(len(Y_pred)):
    ind = int(Y_test[i])
    if ind in coupling_coefficients_value.keys():
        coupling_coefficients_value[ind] = coupling_coefficients_value[ind] + Y_pred[i]
        count[ind] = count[ind] + 1
    if ind not in coupling_coefficients_value.keys():
        coupling_coefficients_value[ind] = Y_pred[i]
        count[ind] = 1

total = np.zeros((num_classes,16))

plt.figure(figsize=(20,9))
for i in range(num_classes):
    average = coupling_coefficients_value[i]/count[i]
    Lindex = i + 1
    plt.subplot(2,4,Lindex)
    plt.title(cell_type[i])
    total[i] = average[i]
    df = DataFrame(np.asmatrix(average),columns=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    if Lindex==1 or Lindex==5:
        heatmap = sns.heatmap(df, yticklabels=['B cells','CD14+ Monocytes','CD4+ T cells','CD8+ T cells','Dendritic Cells',
 'FCGR3A+ Monocytes','Megakaryocytes','Natural killer cells'])
    else:
        heatmap = sns.heatmap(df,yticklabels=[])
plt.show()

###################################################################################################
#overall heatmap
cell_type = ['B cells (1)','CD14+ Monocytes (2)','CD4+ T cells (3)','CD8+ T cells (4)','Dendritic Cells (5)',
 'FCGR3A+ Monocytes (6)','Megakaryocytes (7)','Natural killer cells (8)']

plt.figure()
df = DataFrame(np.asmatrix(total),index=cell_type,columns=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
heatmap = sns.heatmap(df)

plt.ylabel('Type capsule', fontsize=10)
plt.xlabel('Primary capsule', fontsize=10)
plt.show()

#FIG3
##########################################################################################################
#module to plot

pcaB = PCA(n_components=2)
pcaB.fit(encoder10weight0)
weightpcaB = pcaB.transform(encoder10weight0)

pcaCD14 = PCA(n_components=2)
pcaCD14.fit(encoder1weight0)
weightpcaCD14 = pcaCD14.transform(encoder1weight0)

pcaCD4 = PCA(n_components=2)
pcaCD4.fit(encoder2weight0)
weightpcaCD4 = pcaCD4.transform(encoder2weight0)

pcaCD8 = PCA(n_components=2)
pcaCD8.fit(encoder4weight0)
weightpcaCD8 = pcaCD8.transform(encoder4weight0)

pcaDC = PCA(n_components=2)
pcaDC.fit(encoder8weight0)
weightpcaDC = pcaDC.transform(encoder8weight0)

pcaFmono = PCA(n_components=2)
pcaFmono.fit(encoder14weight0)
weightpcaFmono = pcaFmono.transform(encoder14weight0)

pcaMega = PCA(n_components=2)
pcaMega.fit(encoder6weight0)
weightpcaMega = pcaMega.transform(encoder6weight0)

pcaNK = PCA(n_components=2)
pcaNK.fit(encoder16weight0)
weightpcaNK = pcaNK.transform(encoder16weight0)

weight = [weightpcaB, weightpcaCD14,weightpcaCD4,weightpcaCD8,weightpcaDC,weightpcaFmono,weightpcaMega,weightpcaNK]

removed1 = []
removed2 = []
removed3 = []
removed4 = []
removed5 = []
removed6 = []
removed7 = []
removed8 = []

for i in range(3346):
    if weightpcaCD4[i, 0] < -0.046:
        removed3.append(i)
    if weightpcaCD8[i, 0] >0.0845:
        removed4.append(i)
    if weightpcaDC[i, 0] > 0.045:
        removed5.append(i)
    if weightpcaNK[i, 0] > 0.0805:
        removed8.append(i)
    if weightpcaCD14[i, 0] < -0.067:
        removed2.append(i)
    if weightpcaB[i, 0] > 0.06:
        removed1.append(i)
    if weightpcaFmono[i, 0] > 0.0587:
        removed6.append(i)
    if weightpcaMega[i, 1] > 0.0266:
        removed7.append(i)
remove = [removed1,removed2,removed3,removed4,removed5,removed6,removed7,removed8]

mix = [3184,3122,212,446,2729,41,255,816,967,2556]
specific = list(np.asarray(mix) -1)

color = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
label_test = ['NKG7','CD79A','S100A9','CD8A','CCR10','ID3','FCER1A','PF4','CD14','CD19']  #'CD3D',
title = ['Primary capsule 10 (B cells)','Primary capsule 1 (CD14+ Monocytes)', 'Primary capsule 2 (CD4+ T cells)','Primary capsule 4 (CD8+ T cells)',
         'Primary capsule 8 (Dendritic cells)','Primary capsule 14 (FCGR3A+ Monocytes)', 'Primary capsule 6 (Megakaryocytes)',
         'Primary capsule 16 (Natural killer cells)']

plt.figure(figsize=(18,8))

subplotindex = 0
for i in range(8):
    subplotindex = subplotindex + 1
    plt.subplot(2, 4, subplotindex)
    weightlocal = weight[i]
    plt.scatter(weightlocal[:, 0], weightlocal[:, 1], color='r', s=1, alpha=0.1)
    plt.scatter(weightlocal[remove[i], 0], weightlocal[remove[i], 1], color='b', s=4, alpha=0.1)

    for j in range(len(label_test)):
        plt.scatter(weightlocal[specific[j], 0], weightlocal[specific[j], 1], color=color[j], label=label_test[j],marker='*', s=30)

    index = 0
    for x, y in zip(weightlocal[specific, 0], weightlocal[specific, 1]):
        plt.annotate(
            label_test[index],
            xy=(x, y),
            xytext=(0, -3),
            textcoords='offset points',
            ha='center',
            va='top',
            weight = 'ultralight',
            color = color[index]
        )
        index = index + 1

    plt.ylabel('PC2', fontsize=10)
    plt.xlabel('PC1', fontsize=10)
    plt.title(title[i])
plt.show()



