from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

import tensorflow as tf

from inception.slim import slim

FLAGS = tf.app.flags.FLAGS

# If a model is trained using multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

# Batch normalization. Constant governing the exponential moving average of
# the 'global' mean and variance for all activations.
BATCHNORM_MOVING_AVERAGE_DECAY = 0.9997

# The decay to use for the moving average.
MOVING_AVERAGE_DECAY = 0.9999


def inference(images, num_classes, num_of_exs,for_training=False, restore_logits=True,
              scope=None):
  # Parameters for BatchNorm.
  batch_norm_params = {
      # Decay for the moving averages.
      'decay': BATCHNORM_MOVING_AVERAGE_DECAY,
      # epsilon to prevent 0s in variance.
      'epsilon': 0.001,
  }
  # Set weight_decay for weights in Conv and FC layers.
  with slim.arg_scope([slim.ops.conv2d, slim.ops.fc], weight_decay=0.00004):
    with slim.arg_scope([slim.ops.conv2d],
                        activation=tf.nn.relu,
                        batch_norm_params=batch_norm_params):
      logits, dssm, endpoints = slim.nin_dssm.nin_dssm(
          images,
          num_classes=num_classes,
          num_of_exs=num_of_exs,
          is_training=for_training,
          restore_logits=restore_logits,
          scope=scope)

  # Add summaries for viewing model statistics on TensorBoard.
  _activation_summaries(endpoints)

  # Grab the logits associated with the side head. Employed during training.
  # auxiliary_logits = endpoints['aux_logits']

  # TODO add endpoints for extract features.
  return [logits,dssm],endpoints


#TODO inputs=[3*batch_size],dssm=[batch] :)
#TODO add linear svm loss
def loss(logits, labels, batch_size=None):
  """
  thers logits include [logits and dssm(batch*(nums-1))
  """
  num_exs = logits[1].get_shape()[-1].value
  batch_size = FLAGS.batch_size * (num_exs + 1)

  # Reshape the labels into a dense Tensor of
  # shape [FLAGS.batch_size, num_classes].
  #TODO can use tf.ont_hot :)
  sparse_labels = tf.reshape(labels, [batch_size, 1])
  indices = tf.reshape(tf.range(batch_size), [batch_size, 1])
  concated = tf.concat(1, [indices, sparse_labels])
  num_classes = logits[0].get_shape()[-1].value
  dense_labels = tf.sparse_to_dense(concated,
                                    [batch_size, num_classes],
                                    1.0, 0.0)

  # Cross entropy loss for the main softmax prediction.
  #[sim+,sim-,sim-,...]==>[1,0,0,...]
  #TODO using similarity but not distance
  slim.losses.cross_entropy_loss(logits[0],
                                 dense_labels,
                                 label_smoothing=0.1,
                                 weight=0.4)
  # define constant_labels [1,0,0,0,0...]
  labels_ = tf.zeros(FLAGS.batch_size)
  constant_labels = tf.one_hot(labels_,num_exs)
  #TODO should rewrite triplet_loss....(which differ from hinge_loss)
  #TODO there add DSSM loss
  slim.losses.hinge_loss(logits[1],
                        constant_labels,
                        label_smoothing=0.1,
                        weight=1.0)

def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measure the sparsity of activations.

  Args:
    x: Tensor
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.histogram_summary(tensor_name + '/activations', x)
  tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _activation_summaries(endpoints):
  with tf.name_scope('summaries'):
    for act in endpoints.values():
      _activation_summary(act)
