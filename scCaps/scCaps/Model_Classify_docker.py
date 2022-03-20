#timer
import datetime
time_start = datetime.datetime.now()
print('start time:',time_start)

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
import scanpy as sc
import argparse

# configuration
parser = argparse.ArgumentParser(description='scCapsNet')
# system config
parser.add_argument('--inputdata', type=str, default='scClass_data/PBMC68k_modelC.h5ad', help='address for input data')
args = parser.parse_args()
inputdata = args.inputdata
test_name = inputdata.split('/')[1].split('.')[0]

randoms = 30
adata = sc.read_h5ad(inputdata)
data = adata.X.toarray()
labels = adata.obs['modelC id'].values
num_classes = 11
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
x = Reshape((16, z_dim))(x)
capsule = Capsule(num_classes, 16, 3, False)(x)
output = Lambda(lambda x: K.sqrt(K.sum(K.square(x), 2)), output_shape=(num_classes,))(capsule)  
model = Model(inputs=x_in, outputs=output)
model.compile(loss=lambda y_true,y_pred: y_true*K.relu(0.9-y_pred)**2 + 0.25*(1-y_true)*K.relu(y_pred-0.1)**2,\
              optimizer='adam',\
              metrics=['accuracy'])
model.load_weights('scClass_data/Modelweight.weight')
print("I'm still running...")
Y_pred = model.predict(data)
print(Y_pred)
Y_pred_max = np.argmax(Y_pred,axis=1)
print(Y_pred_max)
out = np.vstack([labels,Y_pred_max]).T
np.savetxt('scClass_data/y_pred_'+test_name+'.csv',out,fmt='%i',delimiter=',')

# timer
time_end = datetime.datetime.now()
print('-------------Time Record---------------')
print('start time:',time_start)
print('  end time:',time_end)
print('---------Run time----------')
delta = time_end-time_start
print(delta.seconds//60,'min,',delta.seconds%60,'sec')
