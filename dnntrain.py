#!/usr/bin/env python

# Matplotlib with no Backend
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

# Basic Imports
import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
from time import time

# Keras Imports
from keras.models import Model, load_model
from keras.layers import Input, Dense, Lambda
from keras.layers import Flatten, Reshape, BatchNormalization
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
from keras import regularizers
from keras import backend as K
from keras.callbacks import TensorBoard

# Other Imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from utils import load_data, plot_log, run_tests

# Global Parameters
path = sys.argv[-1]


# Load data
filename = path + '/train.csv'
x_train,y_train=load_data(filename,prints=True,subset=False)
scaler  = MinMaxScaler(feature_range=(-1,1)).fit(y_train.reshape(-1,1))
y_train = scaler.transform(y_train.reshape(-1,1)).squeeze()
x_train,x_valid,y_train,y_valid = train_test_split(x_train,y_train,test_size=0.01,random_state=32)

# ## Training Params
# num_batch = 26
# num_epoch = 30
# patience  = 10

# # Parameters
# ndim = x_train.shape[1]
# L1   = 148
# L2   = 142
# L4   = 23

## Training Params
num_batch = 22
num_epoch = 15
patience  = 10

# Parameters
ndim = x_train.shape[1]
L1   = 127
L2   = 169
L4   = 16


# Model 2
# {'m_units': 172, 'o_units': 20, 'batch_size': 27, 'i_units': 127}
# {'m_units': 169, 'o_units': 16, 'batch_size': 22, 'i_units': 178}


# Build DNN
def build_model():
    x = Input(shape=(ndim,),name='Input')
    h = Dense(L1,activation='relu',name='L1')(x)
    h = BatchNormalization()(h)
    h = Dense(L2,activation='relu',kernel_regularizer=regularizers.l2(0.01),name='L2')(h)
    h = BatchNormalization()(h)
    # h = Dense(L3,activation='relu',kernel_regularizer=regularizers.l2(0.01),name='L3')(h)
    # h = BatchNormalization()(h)
    h = Dense(L4,activation='relu',name='L4')(h)
    h = BatchNormalization()(h)
    y = Dense(1,activation='linear',name='Output')(h)
    model = Model(x,y)
    return model


# Compile Model
model = build_model()
model.summary()
model.compile(optimizer='adam',loss='mean_squared_error')


# Log Files
modelname = path + '/model.h5'
logname   = path + '/log.csv'
figname   = path + '/fig.png'
tplotname = path + '/tplot.png'

# Callback Functions
csv_logger = CSVLogger(logname)
# e_stopper  = EarlyStopping(monitor='val_loss', min_delta=0, patience=patience)
# checkpoint = ModelCheckpoint(modelname,monitor='val_loss',verbose=1,save_best_only=True)

# Train Model
then = time()
log = model.fit(x_train, y_train,
                batch_size=num_batch,
                epochs=num_epoch,
                shuffle=True,
                validation_data=(x_valid,y_valid),
                callbacks = [csv_logger],
                # callbacks=[csv_logger,e_stopper,checkpoint],
                verbose=1)

now = time()
d = divmod(now-then,86400)  # days
h = divmod(d[1],3600)       # hours
m = divmod(h[1],60)         # minutes
s = m[1]                    # seconds
print '\nTotal training time: %02dd:%02dh:%02dm:%02ds'% (d[0],h[0],m[0],s)
print ""
print 'Saving log curves..'
plot_log(log,imgname=figname)


# Run test
# run_tests(path,model,tplotname)

# Test Files
tfiles = [fname for fname in os.listdir(path+'/') if fname.startswith("test")]
fig, axs = plt.subplots(len(tfiles),figsize=(8,6))

for i, file in enumerate(tfiles):
    fname = path + '/' + file
    x_test,y_test=load_data(fname)
    y_test = scaler.transform(y_test.reshape(-1,1)).squeeze()
    y_predict = model.predict(x_test)
    loss = model.evaluate(x_test,y_test,verbose=0)
    
    xx = range(len(y_test))

    if len(tfiles)==1 :
        ax = axs
    else:
        ax = axs[i]

    ax.plot(xx,y_test,xx,y_predict,'g')
    header = '{0}, mse={1:.02f}'.format(fname[9:-4],loss)
    ax.set_title(header)

    oname = path + '/ypred{}.csv'.format(i+1)
    tname = path + '/ytest{}.csv'.format(i+1)
    np.savetxt(oname,y_predict,delimiter=',')
    np.savetxt(tname,y_test,delimiter=',')

plt.legend(['Original','Predicted'],loc=1)
plt.tight_layout()
fig.savefig(tplotname)




