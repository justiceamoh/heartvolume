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
from utils import plot_log

# Global Parameters
path = sys.argv[-1]

# Data load function
def load_data(filename,prints=False,subset=False):
    data   = pd.read_csv(filename,header=None)
    if subset:
        data = data.sample(frac=0.2,random_state=32).reset_index(drop=True)
    else:
        data = data.sample(frac=1,random_state=32).reset_index(drop=True)    
    x_data = data.iloc[:,:-1].values
    y_data = data.iloc[:,-1].values
    if prints:
        print 'X shape:', x_data.shape
        print 'Y shape:', y_data.shape
    return x_data,y_data

# Load data
filename = path + '/train.csv'
x_train,y_train=load_data(filename,True,True)
x_train,x_valid,y_train,y_valid = train_test_split(x_train,y_train,test_size=0.33,random_state=32)

## Training Params
num_batch = 10
num_epoch = 1#30

# Parameters
ndim = x_train.shape[1]
L1   = 256
L2   = 256
L3   = 256
L4   = 32

# Build DNN
def build_model():
    x = Input(shape=(ndim,),name='Input')
    h = Dense(L1,activation='relu',name='L1')(x)
    h = BatchNormalization()(h)
    h = Dense(L2,activation='relu',kernel_regularizer=regularizers.l2(0.01),name='L2')(h)
    h = BatchNormalization()(h)
    h = Dense(L3,activation='relu',kernel_regularizer=regularizers.l2(0.01),name='L3')(h)
    h = BatchNormalization()(h)
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

# Callback Functions
csv_logger = CSVLogger(logname)
e_stopper  = EarlyStopping(monitor='val_loss', min_delta=0, patience=5)
checkpoint = ModelCheckpoint(modelname,monitor='val_loss',verbose=1,save_best_only=True)

# Train Model
then = time()
log = model.fit(x_train, y_train,
                batch_size=num_batch,
                epochs=num_epoch,
                shuffle=True,
                validation_data=(x_valid,y_valid),
                callbacks=[csv_logger,e_stopper,checkpoint],
                verbose=1)

now = time()
d = divmod(now-then,86400)  # days
h = divmod(d[1],3600)       # hours
m = divmod(h[1],60)         # minutes
s = m[1]                    # seconds
print '\nTotal training time: %02dd:%02dh:%02dm:%02ds'% (d[0],h[0],m[0],s)
print ""
print 'Saving log curves..'
# plot_log(log,imgname=figname)




