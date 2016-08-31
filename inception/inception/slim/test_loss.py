#AC
import numpy as np
import tensorflow as tf
from losses import *

def naive_triplet_loss(a,p,n,alpha=1.0):
    """
    :a,p,n 2-D Tensor
    """
    pos_dist = np.sum(np.square(a-p),1)
    neg_dist = np.sum(np.square(a-n),1)
    return np.mean(np.maximum(0.0,alpha+pos_dist-neg_dist),0)
def ex_triplet_loss(a,p,n,alpha=1.0):
    return (naive_triplet_loss(a,p,n,1.0)+naive_triplet_loss(p,a,n,1.0))/2.0

anchor = np.random.randn(4,128)
positive = np.random.randn(4,128)
negative = np.random.randn(4,128)
print "naive loss of np is {}".format(naive_triplet_loss(anchor,positive,negative))
print "triplet loss of np is {}".format(ex_triplet_loss(anchor,positive,negative))

a = tf.placeholder(tf.float32,[4,128])
p = tf.placeholder(tf.float32,[4,128])
n = tf.placeholder(tf.float32,[4,128])

loss = triplet_ex_loss(a,p,n,alpha=1.0)
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    feed={a:anchor,p:positive,n:negative}
    rst = sess.run(loss,feed)
    print "triplet loss of tf is {}".format(rst)
