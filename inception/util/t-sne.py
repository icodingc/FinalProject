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
from sklearn.preprocessing import normalize
from tsne import bh_sne
root_name = '/home/zhangxuesen/workshops/FinalProject2/inception/features/'

feats = np.load(os.path.join(root_name,'triplet/file_features.npy')).astype(np.float64)
feats = normalize(np.squeeze(feats))
labels = np.loadtxt(os.path.join(root_name,'labels.lst'))-1.0 
X_2d = bh_sne(feats)
fig = plt.figure()
plt.scatter(X_2d[:,0],X_2d[:,1],c=labels)
fig.savefig('cifar10_triplet.pdf')

