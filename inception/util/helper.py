"""
mAP for ranking based criterion
"""
import numpy as np
from scipy.spatial import distance
import os,sys
from tqdm import tqdm

root_dir = 'home/zhangxuesen/workshops/FinalProject2/inception/features/'
with open(os.path.join(root_dir,'filenames.lst')) as f:
    names = f.readlines()
names = [a.strip() for a in names]
with open(os.path.join(root_dir,'labels.lst')) as f:
    labels = f.readlines()
labels = [int(a.strip()) for a in labels]

features = np.load(os.path.join(root_dir,'xent/file_features.npy'))
name2label = dict(zip(names,labels))
name2idx = dict(zip(names,list(xrange(10000))))

with open(os.path.join(root_dir,'query.lst')) as f:
    querys = f.readlines()
querys = [a.strip() for a in querys]

with open(os.path.join(root_dir,'query_labels.lst')) as f:
    query_labelss = f.readlines()

def name2feat(p):
    return features[name2idx[p]]

def Rei(q,i):
    """
    :q query image [name.jpg]
    :i i-th ranked image [name.jpg]
    :return {0,1}
    """
    return int(name2label[q]==name2label[i])
def AP(q,k):
    """
    :q query image [name.jpg]
    :k top@K
    """
    feat = name2feat(q)
    #TODO add annoy 
    dist = np.array([distance.euclidean(a,feat) for a in features])
    idx = np.argsort(dist)[:k]
    return sum([Rei(q,names[i]) for i in idx])/(k*1.0)
def mAP(ps,k):
    """
    :list of query name.jpg
    """
    cnt = len(ps)
    return sum([AP(p,k) for p in tqdm(ps)])/(cnt*1.0)

print(mAP(querys,k=10))
