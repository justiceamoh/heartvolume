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

# Hyperopt
from datetime import datetime
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials 
from hyperopt import space_eval
from hyperopt.pyll.base import scope

# Other Imports
from sklearn.model_selection import train_test_split
from utils import plot_log


#################################
## HYPEROPT DEFINITIONS        ##
#################################
space = { 'i_units': scope.int(hp.uniform('units1', 50,500)),
          'm_units': scope.int(hp.uniform('units2',100,500)),
          'o_units': scope.int(hp.uniform('units3', 16,64)),

          # 'dropout1': hp.uniform('dropout1',.2,.5),
          # 'dropout2': hp.uniform('dropout2',.2,.5),

          'batch_size' : scope.int(hp.uniform('batch_size', 10, 30)),
          'layers': hp.choice('layers',['Five','Four']),
          # 'optimizer': hp.choice('optimizer',['adam','rmsprop']),
          # 'loss': hp.choice('loss',['categorical_crossentropy','squared_hinge','mean_squared_error']),
        }



# Global Parameters
path = sys.argv[1]

# Data load function
def load_data(filename,prints=False):
    data   = pd.read_csv(filename,header=None)
    x_data = data.iloc[:,:-1].values
    y_data = data.iloc[:,-1].values
    if prints:
        print 'X shape:', x_data.shape
        print 'Y shape:', y_data.shape
    return x_data,y_data

# Load data
filename = path + '/train.csv'
x_train,y_train=load_data(filename)
x_train,x_valid,y_train,y_valid = train_test_split(x_train,y_train,test_size=0.33,random_state=32)

def f_nn(params):
    ## Training Params
    num_batch = params['batch_size']
    num_epoch = 1#30

    addlayer  = params['layers'] == 'Five'

    # Parameters
    ndim = x_train.shape[1]
    L1   = params['i_units']
    L2   = params['m_units']
    L3   = params['m_units']
    L4   = params['o_units']

    # Build DNN
    def build_model():
        x = Input(shape=(ndim,),name='Input')
        h = Dense(L1,activation='relu',name='L1')(x)
        h = BatchNormalization()(h)
        h = Dense(L2,activation='relu',kernel_regularizer=regularizers.l2(0.01),name='L2')(h)
        h = BatchNormalization()(h)
        if addlayer:
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

    modelname  = 'hypermodel.h5'
    e_stopper  = EarlyStopping(monitor='val_loss', min_delta=0, patience=5)
    checkpoint = ModelCheckpoint(modelname,monitor='val_loss',verbose=1,save_best_only=True)

    # Train Model
    log = model.fit(x_train, y_train,
                    batch_size=num_batch,
                    epochs=num_epoch,
                    shuffle=True,
                    validation_data=(x_valid,y_valid),
                    callbacks=[e_stopper,checkpoint],
                    verbose=1)

    model = load_model(modelname)
    loss = model.evaluate(x_valid,y_valid,verbose=0)
    return {'loss': loss, 'status': STATUS_OK}


#################################
## HYPERPARAMETER SEARCH       ##
#################################
trials = Trials()
best   = fmin(f_nn, space, algo=tpe.suggest, max_evals=25, trials=trials)

# Get the values of the optimal parameter
best_params = space_eval(space, best)
print 'best: '
print best_params

# Save the hyperparameter at each iteration to a csv file
param_values = [x['misc']['vals'] for x in trials.trials]
param_values = [{key:value for key in x for value in x[key]} for x in param_values]
param_values = [space_eval(space, x) for x in param_values]

param_df = pd.DataFrame(param_values)
param_df['MSE'] = [1 - x for x in trials.losses()]
param_df.index.name = 'Iteration'
param_df.to_csv("hyperparameters.csv")

