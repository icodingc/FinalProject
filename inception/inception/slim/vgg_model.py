from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from inception.slim import ops
from inception.slim import scopes


def vgg(inputs,
        num_classes=1000,
        is_training=True,
        restore_logits=True,
        scope=''):
  # end_points will collect relevant activations for external use, for example
  # summaries or losses.
  end_points = {}
  with tf.op_scope([inputs], scope, 'vgg'):
    with scopes.arg_scope([ops.conv2d, ops.fc, ops.batch_norm],
                          is_training=is_training):
        # conv1
        end_points['conv1'] = ops.repeat_op(2,inputs,ops.conv2d,64,[3,3],scope='conv1')
        end_points['pool1'] = ops.max_pool(end_points['conv1'],[2,2],scope='pool1')
        # conv2
        end_points['conv2'] = ops.repeat_op(2,end_points['pool1'],ops.conv2d,128,[3,3],scope='conv2')
        end_points['pool2'] = ops.max_pool(end_points['conv2'],[2,2],scope='pool2')
        # conv3
        end_points['conv3'] = ops.repeat_op(2,end_points['pool2'],ops.conv2d,256,[3,3],scope='conv3')
        end_points['pool3'] = ops.max_pool(end_points['conv3'],[2,2],scope='pool3')
        # conv4
        end_points['conv4'] = ops.repeat_op(2,end_points['pool3'],ops.conv2d,512,[3,3],scope='conv4')
        end_points['pool4'] = ops.max_pool(end_points['conv4'],[2,2],scope='pool4')
        # conv5
        end_points['conv5'] = ops.repeat_op(2,end_points['pool4'],ops.conv2d,512,[3,3],scope='conv5')
        end_points['pool5'] = ops.max_pool(end_points['conv5'],[2,2],scope='pool5')
        
        end_points['flatten5'] = ops.flatten(end_points['pool5'],scope='flatten5')
        end_points['fc6'] = ops.fc(end_points['flatten5'],4096,scope='fc6')
        end_points['dropout6'] = ops.dropout(end_points['fc6'],0.5,scope='dropout6')
        end_points['fc7'] = ops.fc(end_points['dropout6'],4096,scope='fc7')
        end_points['dropout7'] = ops.dropout(end_points['fc7'],0.5,scope='dropout7')

        logits = ops.fc(end_points['fc7'],num_classes,activation=None,scope='fc8')
    return logits, end_points
