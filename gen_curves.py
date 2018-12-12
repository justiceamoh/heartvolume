#!/usr/bin/env python

# Author: Justice Amoh
# Date: 03/26/2018
# Description: Script for Generating Aggregated Curves for Crossvalidation Experiments

import os
import sys
import re
import numpy as np
import pandas as pd

# Matplotlib with no Backend
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from glob import glob 
import seaborn as sns
sns.set_style("whitegrid")


root  = 'xval_logs'

# Load All CSVs in folder
dfs = []

query = root + '/*.csv'
csvs  = glob(query)
for csv in csvs:
    df = pd.read_csv(csv)
    df['fold'] = int(re.search(r"(?<=Fold).*?(?=_)",csv).group(0))
    dfs.append(df)
        
df = pd.concat(dfs,ignore_index=True)

# Generate Curves
fig, axs = plt.subplots(nrows=1,ncols=2,figsize=(12,6))
fig.suptitle('10-Fold XVal Curves',fontsize=14)
sns.tsplot(time='epoch', value='loss',unit='fold', 
           data=df,ax=axs[0])
sns.tsplot(time='epoch', value='val_loss',unit='fold', 
           data=df,ax=axs[1])
axs[0].set_title('Training Loss')
axs[1].set_title('Validation Loss')
# axs[0].set_xlim([0,20])
# axs[1].set_xlim([0,20])

outfile = 'res_curves.png'
fig.savefig(outfile)

print 'File saved as %s.'%outfile