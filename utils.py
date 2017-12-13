#!/usr/bin/env python

# Author: Justice Amoh
# Date: 12/13/2017
# Description: Utility functions for training nets

# Matplotlib with no Backend
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


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
