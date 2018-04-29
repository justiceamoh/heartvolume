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
from keras.layers import Input, Dense, Lambda, AveragePooling2D
from keras.layers import Flatten, Reshape, BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
from keras import regularizers
from keras import backend as K
from keras.callbacks import TensorBoard


# Other Imports
from time import time
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from utils import plot_log



# Global Parameters
path = sys.argv[1]

# Data load function
def load_data(filename,prints=False):
    data   = pd.read_csv(filename,header=None)
    x_data = data.iloc[:,:-1].values.reshape((-1,30,224,1))
    y_data = data.iloc[:,-1].values
    if prints:
        print 'X shape:', x_data.shape
        print 'Y shape:', y_data.shape
    return x_data,y_data

# Load data
filename = path + '/train.csv'
x_data,y_data=load_data(filename)
scaler   = MinMaxScaler(feature_range=(0,1)).fit(y_data.reshape(-1,1))
y_data   = scaler.transform(y_data.reshape(-1,1)).squeeze()
# x_train,x_valid,y_train,y_valid = train_test_split(x_train,y_train,test_size=0.33,random_state=32)


## Training Params
K         = 10
num_batch = 28
num_epoch = 30
patience  = 10


# Parameters
npat,ndim,_ = x_data.shape[1:]
L1   = 28
L2   = 16
L3   = 15
L4   = 7



# Build CNN
def build_model():
    x = Input(shape=(npat,ndim,1),name='Input')
    h = Conv2D(L1,kernel_size=(3,5),strides=(2,3),padding='same',name='L1')(x)
    h = BatchNormalization()(h)
    h = Conv2D(L2,kernel_size=(3,3),strides=(2,3),padding='same',name='L2')(h)
    h = BatchNormalization()(h)
    h = Conv2D(L3,kernel_size=(3,3),strides=(1,3),padding='same',name='L3')(h)
    h = BatchNormalization()(h)
    h = AveragePooling2D((2,2),strides=(2,2))(h)
    h = Flatten()(h)
    h = Dense(L4,activation='relu',kernel_regularizer=regularizers.l2(0.01),name='L4')(h)
    y = Dense(1,activation='linear',name='Output')(h)
    model = Model(x,y)
    return model


#############################
## K-FOLD X-VALIDATION     ##
#############################

# StratifiedKFold
# skf  = StratifiedKFold(n_splits=K,shuffle=True,random_state=32)
kf   = KFold(n_splits=K, random_state=32, shuffle=True)
then = time()

# for k, (idx_train, idx_valid) in enumerate(skf.split(x_data, y_data)):
for k, (idx_train, idx_valid) in enumerate(kf.split(x_data, y_data)):    
    print '\nFold %d/%d'%(k+1,K)
    print ''

    x_train, x_valid = x_data[idx_train], x_data[idx_valid]
    y_train, y_valid = y_data[idx_train], y_data[idx_valid]

    model = build_model()
    model.compile(optimizer='adam',loss='mean_squared_error')


    logname  = path + '/Fold{:02}_log.csv'.format(k+1)
    csvlogger = CSVLogger(logname)
    # modelname  = 'xvalmodel.h5'
    # e_stopper  = EarlyStopping(monitor='val_loss', min_delta=0, patience=20)
    # checkpoint = ModelCheckpoint(modelname,monitor='val_loss',verbose=1,save_best_only=True)


    # Train Model
    log = model.fit(x_train, y_train,
                    batch_size=num_batch,
                    epochs=num_epoch,
                    shuffle=True,
                    validation_data=(x_valid,y_valid),
                    callbacks=[csvlogger],
                    verbose=2)


    loss = model.evaluate(x_valid,y_valid,verbose=0)
    print 'Loss: {:0.4f}'.format(loss)
    print 'Log saved as: {}'.format(logname)



# Report Time
now = time()
d = divmod(now-then,86400)  # days
h = divmod(d[1],3600)       # hours
m = divmod(h[1],60)         # minutes
s = m[1]                    # seconds
print '\nTotal training time: %02dd:%02dh:%02dm:%02ds'% (d[0],h[0],m[0],s)
print '\n\n'



