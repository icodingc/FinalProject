# resnet: improved scheme proposed in Identity Mappings in Deep Residual Networks 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from inception.slim import ops
from inception.slim import scopes

#TODO (add to slim arg_scope)
def bn_relu(inputs):
    return tf.nn.relu(ops.batch_norm(inputs))

def block34(x,filters,kernel,scope='',stride=1,ex=False):
    net = bn_relu(x)
    net = ops.conv2d(net,filters,[kernel,kernel],stride=stride,scope=scope+'1')
    if ex : x = ops.conv2d(x,filters,[kernel,kernel],stride=stride,activation=None,batch_norm_params=None,scope=scope+'ex') 
    return x + ops.conv2d(net,filters,[kernel,kernel],activation=None,batch_norm_params=None,scope=scope+'2') 

def resnet34(inputs,
        num_classes=1000,
        is_training=True,
        restore_logits=True,
        scope=''):
  end_points = {}
  with tf.op_scope([inputs], scope, 'resnet'):
    with scopes.arg_scope([ops.conv2d,ops.batch_norm],
                          is_training=is_training):
        # 224 x 224 x 3
        end_points['conv1'] = ops.conv2d(inputs, 64, [7, 7], stride=2,scope='conv1')
        end_points['pool1'] = ops.max_pool(end_points['conv1'],[3,3],stride=2,padding='SAME',scope='pool1')
        # 56 * 56
        #TODO (using loop)
        end_points['conv2_1'] = block34(end_points['pool1'],64,3,'res2_1')
        end_points['conv2_2'] = block34(end_points['conv2_1'],64,3,'res2_2')
        end_points['conv2_3'] = block34(end_points['conv2_2'],64,3,'res2_3')
        # 56 * 56
        end_points['conv3_1'] = block34(end_points['conv2_3'],128,3,'res3_1',stride=2,ex=True)
        end_points['conv3_2'] = block34(end_points['conv3_1'],128,3,'res3_2')
        end_points['conv3_3'] = block34(end_points['conv3_2'],128,3,'res3_3')
        end_points['conv3_4'] = block34(end_points['conv3_3'],128,3,'res3_4')
        # 28 * 28
        end_points['conv4_1'] = block34(end_points['conv3_4'],256,3,'res4_1',stride=2,ex=True)
        end_points['conv4_2'] = block34(end_points['conv4_1'],256,3,'res4_2')
        end_points['conv4_3'] = block34(end_points['conv4_2'],256,3,'res4_3')
        end_points['conv4_4'] = block34(end_points['conv4_3'],256,3,'res4_4')
        end_points['conv4_5'] = block34(end_points['conv4_4'],256,3,'res4_5')
        end_points['conv4_6'] = block34(end_points['conv4_5'],256,3,'res4_6')
        # 14 * 14
        end_points['conv5_1'] = block34(end_points['conv4_6'],512,3,'res5_1',stride=2,ex=True)
        end_points['conv5_2'] = block34(end_points['conv5_1'],512,3,'res5_2')
        end_points['conv5_3'] = block34(end_points['conv5_2'],512,3,'res5_3')
        #7 * 7 * 512 
        end_points['avg'] = ops.avg_pool(end_points['conv5_3'],[7,7],stride=1,padding='SAME',scope='avg_pooling')
        end_points['flatten'] = ops.flatten(end_points['avg'],scope='flatten')
        end_points['logits'] = ops.fc(end_points['flatten'],num_classes,scope='logits')

        return end_points['logits'],end_points
def resnet(inputs,is_train=True,scope=''):
	batch_norm_params = {
      	'decay': 0.9997,
      	'epsilon': 0.001,}
	with scopes.arg_scope([ops.conv2d,ops.deconv2d],weight_decay=0.0005,
				stddev=0.1,
				activation=tf.nn.relu,
				batch_norm_params=batch_norm_params):
		return net(inputs,is_train,scope)
