"""
from tsne import bh_sne
x = ...
y = ...
X_2d = bh_sne(X)
fig = plt.figure()
plt.scatter(X_2d[:,0],X_2d[:,1],c=y)
fig.savefig('foo.png')
"""
import numpy as np
import matplotlib.pyplot as plt
import os,sys
from tsne import bh_sne
root_name = '/home/zhangxuesen/workshops/FinalProject2/inception/features/'

feats = np.load(os.path.join(root_name,'xent/file_features.npy')).astype(np.float64)
labels = np.loadtxt(os.path.join(root_name,'labels.lst'))-1.0 
X_2d = bh_sne(feats)
fig = plt.figure()
plt.scatter(X_2d[:,0],X_2d[:,1],c=labels)
fig.savefig('cifar10.pdf')

