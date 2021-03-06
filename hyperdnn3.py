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

# Hyperopt
from datetime import datetime
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials 
from hyperopt import space_eval
from hyperopt.pyll.base import scope

# Other Imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from utils import plot_log


#################################
## HYPEROPT DEFINITIONS        ##
#################################
space = { 'l1_units': scope.int(hp.uniform('units1',6,32)),
          'l2_units': scope.int(hp.uniform('units2',6,32)),
          'l3_units': scope.int(hp.uniform('units3',6,32)),
          'l4_units': scope.int(hp.uniform('units4',5,20)),

          # 'dropout1': hp.uniform('dropout1',.2,.5),
          # 'dropout2': hp.uniform('dropout2',.2,.5),

          'batch_size' : scope.int(hp.uniform('batch_size', 15, 30)),
          # 'layers': hp.choice('layers',['Five','Four']),
          # 'optimizer': hp.choice('optimizer',['adam','rmsprop']),
          # 'loss': hp.choice('loss',['categorical_crossentropy','squared_hinge','mean_squared_error']),
        }



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
x_train,y_train=load_data(filename)
scaler   = MinMaxScaler(feature_range=(0,1)).fit(y_train.reshape(-1,1))
y_train  = scaler.transform(y_train.reshape(-1,1)).squeeze()
x_train,x_valid,y_train,y_valid = train_test_split(x_train,y_train,test_size=0.33,random_state=32)

def f_nn(params):
    ## Training Params
    num_batch = params['batch_size']
    num_epoch = 30


    # Parameters
    npat,ndim,_ = x_train.shape[1:]
    L1   = params['l1_units']
    L2   = params['l2_units']
    L3   = params['l3_units']
    L4   = params['l4_units']

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


    # Compile Model
    model = build_model()
    # model.summary()
    model.compile(optimizer='adam',loss='mean_squared_error')

    modelname  = 'hypermodel.h5'
    e_stopper  = EarlyStopping(monitor='val_loss', min_delta=0, patience=20)
    checkpoint = ModelCheckpoint(modelname,monitor='val_loss',verbose=1,save_best_only=True)

    # Train Model
    log = model.fit(x_train, y_train,
                    batch_size=num_batch,
                    epochs=num_epoch,
                    shuffle=True,
                    validation_data=(x_valid,y_valid),
                    callbacks=[e_stopper,checkpoint],
                    # callbacks=[e_stopper],
                    verbose=2)

    model = load_model(modelname)

    loss = model.evaluate(x_valid,y_valid,verbose=0)
    print '\n\nLoss: {:0.4f}'.format(loss)
    print 'L1: {}, L2: {}, L4: {}'.format(L1,L2,L4)
    print 'BatchSize: {}\n\n'.format(num_batch)
    return {'loss': loss, 'status': STATUS_OK}


#################################
## HYPERPARAMETER SEARCH       ##
#################################
then = time()
trials = Trials()
best   = fmin(f_nn, space, algo=tpe.suggest, max_evals=25, trials=trials)

# Report Time
now = time()
d = divmod(now-then,86400)  # days
h = divmod(d[1],3600)       # hours
m = divmod(h[1],60)         # minutes
s = m[1]                    # seconds
print '\nTotal training time: %02dd:%02dh:%02dm:%02ds'% (d[0],h[0],m[0],s)
print '\n\n'


# Get the values of the optimal parameter
best_params = space_eval(space, best)
print 'best: '
print best_params

# Save the hyperparameter at each iteration to a csv file
param_values = [x['misc']['vals'] for x in trials.trials]
param_values = [{key:value for key in x for value in x[key]} for x in param_values]
param_values = [space_eval(space, x) for x in param_values]

param_df = pd.DataFrame(param_values)
param_df['MSE'] = [x for x in trials.losses()]
param_df.index.name = 'Iteration'
param_df.to_csv("hyperparameters3.csv")

