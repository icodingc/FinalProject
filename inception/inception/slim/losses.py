# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains convenience wrappers for various Neural Network TensorFlow losses.

  All the losses defined here add themselves to the LOSSES_COLLECTION
  collection.

  l1_loss: Define a L1 Loss, useful for regularization, i.e. lasso.
  l2_loss: Define a L2 Loss, useful for regularization, i.e. weight decay.
  cross_entropy_loss: Define a cross entropy loss using
    softmax_cross_entropy_with_logits. Useful for classification.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

# In order to gather all losses in a network, the user should use this
# key for get_collection, i.e:
#   losses = tf.get_collection(slim.losses.LOSSES_COLLECTION)
LOSSES_COLLECTION = '_losses'


def l1_regularizer(weight=1.0, scope=None):
  """Define a L1 regularizer.

  Args:
    weight: scale the loss by this factor.
    scope: Optional scope for op_scope.

  Returns:
    a regularizer function.
  """
  def regularizer(tensor):
    with tf.op_scope([tensor], scope, 'L1Regularizer'):
      l1_weight = tf.convert_to_tensor(weight,
                                       dtype=tensor.dtype.base_dtype,
                                       name='weight')
      return tf.mul(l1_weight, tf.reduce_sum(tf.abs(tensor)), name='value')
  return regularizer


def l2_regularizer(weight=1.0, scope=None):
  """Define a L2 regularizer.

  Args:
    weight: scale the loss by this factor.
    scope: Optional scope for op_scope.

  Returns:
    a regularizer function.
  """
  def regularizer(tensor):
    with tf.op_scope([tensor], scope, 'L2Regularizer'):
      l2_weight = tf.convert_to_tensor(weight,
                                       dtype=tensor.dtype.base_dtype,
                                       name='weight')
      return tf.mul(l2_weight, tf.nn.l2_loss(tensor), name='value')
  return regularizer


def l1_l2_regularizer(weight_l1=1.0, weight_l2=1.0, scope=None):
  """Define a L1L2 regularizer.

  Args:
    weight_l1: scale the L1 loss by this factor.
    weight_l2: scale the L2 loss by this factor.
    scope: Optional scope for op_scope.

  Returns:
    a regularizer function.
  """
  def regularizer(tensor):
    with tf.op_scope([tensor], scope, 'L1L2Regularizer'):
      weight_l1_t = tf.convert_to_tensor(weight_l1,
                                         dtype=tensor.dtype.base_dtype,
                                         name='weight_l1')
      weight_l2_t = tf.convert_to_tensor(weight_l2,
                                         dtype=tensor.dtype.base_dtype,
                                         name='weight_l2')
      reg_l1 = tf.mul(weight_l1_t, tf.reduce_sum(tf.abs(tensor)),
                      name='value_l1')
      reg_l2 = tf.mul(weight_l2_t, tf.nn.l2_loss(tensor),
                      name='value_l2')
      return tf.add(reg_l1, reg_l2, name='value')
  return regularizer


def l1_loss(tensor, weight=1.0, scope=None):
  """Define a L1Loss, useful for regularize, i.e. lasso.

  Args:
    tensor: tensor to regularize.
    weight: scale the loss by this factor.
    scope: Optional scope for op_scope.

  Returns:
    the L1 loss op.
  """
  with tf.op_scope([tensor], scope, 'L1Loss'):
    weight = tf.convert_to_tensor(weight,
                                  dtype=tensor.dtype.base_dtype,
                                  name='loss_weight')
    loss = tf.mul(weight, tf.reduce_sum(tf.abs(tensor)), name='value')
    tf.add_to_collection(LOSSES_COLLECTION, loss)
    return loss


def l2_loss(tensor, weight=1.0, scope=None):
  """Define a L2Loss, useful for regularize, i.e. weight decay.

  Args:
    tensor: tensor to regularize.
    weight: an optional weight to modulate the loss.
    scope: Optional scope for op_scope.

  Returns:
    the L2 loss op.
  """
  with tf.op_scope([tensor], scope, 'L2Loss'):
    weight = tf.convert_to_tensor(weight,
                                  dtype=tensor.dtype.base_dtype,
                                  name='loss_weight')
    loss = tf.mul(weight, tf.nn.l2_loss(tensor), name='value')
    tf.add_to_collection(LOSSES_COLLECTION, loss)
    return loss


def cross_entropy_loss(logits, one_hot_labels, label_smoothing=0,
                       weight=1.0, scope=None):
  """Define a Cross Entropy loss using softmax_cross_entropy_with_logits.

  It can scale the loss by weight factor, and smooth the labels.

  Args:
    logits: [batch_size, num_classes] logits outputs of the network .
    one_hot_labels: [batch_size, num_classes] target one_hot_encoded labels.
    label_smoothing: if greater than 0 then smooth the labels.
    weight: scale the loss by this factor.
    scope: Optional scope for op_scope.

  Returns:
    A tensor with the softmax_cross_entropy loss.
  """
  logits.get_shape().assert_is_compatible_with(one_hot_labels.get_shape())
  with tf.op_scope([logits, one_hot_labels], scope, 'CrossEntropyLoss'):
    num_classes = one_hot_labels.get_shape()[-1].value
    one_hot_labels = tf.cast(one_hot_labels, logits.dtype)
    if label_smoothing > 0:
      smooth_positives = 1.0 - label_smoothing
      smooth_negatives = label_smoothing / num_classes
      one_hot_labels = one_hot_labels * smooth_positives + smooth_negatives
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits,
                                                            one_hot_labels,
                                                            name='xentropy')
    weight = tf.convert_to_tensor(weight,
                                  dtype=logits.dtype.base_dtype,
                                  name='loss_weight')
    loss = tf.mul(weight, tf.reduce_mean(cross_entropy), name='value')
    tf.add_to_collection(LOSSES_COLLECTION, loss)
    return loss
#TODO add triplet_loss
# anchor,positive,negative = tf.split(0,3,logits)
def triplet_loss(anchor,positive,negative,alpha,
                scope=None):
  with tf.op_scope([anchor,positive,negative], scope, 'TripletLoss'):
    alpha = tf.convert_to_tensor(alpha,
                                  dtype=anchor.dtype.base_dtype,
                                  name='alpha_margin')
    pos_dist = tf.reduce_sum(tf.square(tf.sub(anchor,positive)),1)
    neg_dist = tf.reduce_sum(tf.square(tf.sub(anchor,negative)),1)
    basic_loss = tf.nn.relu(tf.add(tf.sub(pos_dist,neg_dist),alpha),name='tripletloss')
    return tf.reduce_mean(basic_loss,0)
#TODO a trick to expand triplets eg. (pos,anc,neg) also a triplet.
def triplet_ex_loss(anchor,positive,negative,alpha,
                scope=None):
  with tf.op_scope([anchor,positive,negative], scope, 'TripletLoss'):
    alpha = tf.convert_to_tensor(alpha,
                                  dtype=anchor.dtype.base_dtype,
                                  name='alpha_margin')
    return (triplet_loss(anchor,positive,negative,alpha)+triplet_loss(positive,anchor,negative,alpha))/2.0
#TODO customized loss [the label always is 0]
def hinge_loss(logits,alpha,nrof_pairs=3,
                scope=None):
  with tf.op_scope([logits], scope, 'HingeLoss'):
    rst = tf.split(0,nrof_pairs,logits)
    assert nrof_pairs>=3
    if nrof_pairs==3:return triplet_loss(rst[0],rst[1],rst[2],alpha,'triplet_loss')

    anchor,positive = rst[0],rst[1]
    return tf.add_n([triplet_loss(anchor,positive,rst[i]) for i in xrange(2,nrof_pairs)])/tf.cast(nrof_pairs-2,tf.float32)
