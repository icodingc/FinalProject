# could also use model.ckpt or export
import numpy as np
import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
root_name = '../trained_model/00074999/'

config = tf.ConfigProto(allow_soft_placement=True)
sess = tf.Session(config=config)
saver = tf.train.import_meta_graph(root_name+'export.meta')
saver.restore(sess,root_name+'export')

inputs = sess.graph.get_tensor_by_name('Placeholder:0')
outputs = sess.graph.get_tensor_by_name('nin/logits_xent/xw_plus_b:0')

img = tf.gfile.FastGFile('test.jpg','r').read()
imgs = np.array([img]*10)
rst = sess.run(outputs,{inputs:imgs})
print type(rst)
print rst.shape
