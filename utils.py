#!/usr/bin/env python

# Author: Justice Amoh
# Date: 12/13/2017
# Description: Utility functions for training nets

import os
import pandas as pd
import numpy as np 
# Matplotlib with no Backend
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

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

def plot_log(log,imgname='trainlog.png'):
    if isinstance(log,str):
        df    = pd.read_csv(log)
        epchs = df['epoch'].values
        tloss = df['loss'].values
        vloss = df['val_loss'].values
    else:
        epchs = log.epoch
        tloss = log.history['loss']
        vloss = log.history['val_loss']

    fig, axs = plt.subplots(figsize=(5,4),sharex=True)
    axs.plot(epchs,tloss,epchs,vloss)
    axs.set_ylabel('Loss')
    axs.set_title('Training Curves')
    axs.legend(['Train','Valid'])    
    # axr[1].xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))

    fig.savefig(imgname)
    print 'Log curves saved as: ' + imgname
    return

def run_tests(path,model,imgname='testplots.png'):
    # Test Files
    tfiles = [fname for fname in os.listdir(path+'/') if fname.startswith("test")]
    fig, axs = plt.subplots(len(tfiles),figsize=(5,6))

    for i, file in enumerate(tfiles):
        fname = path + '/' + file
        x_test,y_test=load_data(fname)
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
    fig.savefig(imgname)
