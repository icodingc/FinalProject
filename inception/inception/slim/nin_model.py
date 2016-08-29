from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import tensorflow as tf

from inception.slim import ops
from inception.slim import scopes


def nin(inputs,
        num_classes=10,
        is_training=True,
        restore_logits=True,
        scope=''):
  # end_points will collect relevant activations for external use, for example
  # summaries or losses.
  end_points = {}
  with tf.op_scope([inputs], scope, 'nin'):
    with scopes.arg_scope([ops.conv2d, ops.fc, ops.batch_norm],
                          is_training=is_training):
        # conv1
        end_points['conv1'] = ops.conv2d(inputs,192,[5,5],scope='conv1')
        end_points['conv1_1'] = ops.conv2d(end_points['conv1'],160,[1,1],scope='conv1_1')
        end_points['conv1_2'] = ops.conv2d(end_points['conv1_1'],96,[1,1],scope='conv1_2')
        end_points['pool1'] = ops.max_pool(end_points['conv1_2'],[3,3],stride=2,
                padding='SAME',scope='pool1')
        net = ops.dropout(end_points['pool1'],0.5)
        # conv2
        end_points['conv2'] = ops.conv2d(net,192,[5,5],scope='conv2')
        end_points['conv2_1'] = ops.conv2d(end_points['conv2'],192,[1,1],scope='conv2_1')
        end_points['conv2_2'] = ops.conv2d(end_points['conv2_1'],192,[1,1],scope='conv2_2')
        end_points['pool2'] = ops.max_pool(end_points['conv2_2'],[3,3],stride=2,
                padding='SAME',scope='pool2')
        net = ops.dropout(end_points['pool2'],0.5)
        # conv3
        end_points['conv3'] = ops.conv2d(net,192,[3,3],scope='conv3')
        end_points['conv3_1'] = ops.conv2d(end_points['conv3'],192,[1,1],scope='conv3_1')
        end_points['conv3_2'] = ops.conv2d(end_points['conv3_1'],10,[1,1],scope='conv3_2')
        net = ops.avg_pool(end_points['conv3_2'],[8,8],scope='avg_pool')
        flatten = ops.flatten(net,scope='flatten')
        #TODO take care this,using num_classes but 10..
        end_points['logits'] = ops.fc(flatten,num_classes,activation=None,scope='fc')

    return end_points['logits'],end_points
