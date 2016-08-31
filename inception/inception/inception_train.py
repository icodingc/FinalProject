from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
from datetime import datetime
import os.path
import re
import time

import numpy as np
import tensorflow as tf

from inception import image_processing
from inception import inception_model as inception
#from inception import nin_model as inception
from inception.slim import slim

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/imagenet_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 10000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_string('subset', 'train',
                           """Either 'train' or 'validation'.""")

# Flags governing the hardware employed for running TensorFlow.
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            """How many GPUs to use.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

# Flags governing the type of training.
tf.app.flags.DEFINE_boolean('fine_tune', False,
                            """If set, randomly initialize the final layer """
                            """of weights in order to train the network on a """
                            """new task.""")
tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', '',
                           """If specified, restore this pretrained model """
                           """before beginning any training.""")

# **IMPORTANT**
# Please note that this learning rate schedule is heavily dependent on the
# hardware architecture, batch size and any changes to the model architecture
# specification. Selecting a finely tuned learning rate schedule is an
# empirical process that requires some experimentation. Please see README.md
# more guidance and discussion.
#
# With 8 Tesla K40's and a batch size = 256, the following setup achieves
# precision@1 = 73.5% after 100 hours and 100K steps (20 epochs).
# Learning rate decay factor selected from http://arxiv.org/abs/1404.5997.
tf.app.flags.DEFINE_float('initial_learning_rate', 0.1,
                          """Initial learning rate.""")
tf.app.flags.DEFINE_float('num_epochs_per_decay', 30.0,
                          """Epochs after which learning rate decays.""")
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.16,
                          """Learning rate decay factor.""")

# Constants dictating the learning rate schedule.
RMSPROP_DECAY = 0.9                # Decay term for RMSProp.
RMSPROP_MOMENTUM = 0.9             # Momentum in RMSProp.
RMSPROP_EPSILON = 1.0              # Epsilon term for RMSProp.


def _total_loss(images, labels, output_dims, scope):
  """Calculate the total loss on a single tower running the ImageNet model.

  We perform 'batch splitting'. This means that we cut up a batch across
  multiple GPU's. For instance, if the batch size = 32 and num_gpus = 2,
  then each tower will operate on an batch of 16 images.

  Args:
    images: Images. 4D tensor of size [batch_size, FLAGS.image_size,
                                       FLAGS.image_size, 3].
    labels: 1-D integer Tensor of [batch_size].
    output_dims. [int,int]
    scope: unique prefix string identifying the ImageNet tower, e.g.
      'tower_0'.

  Returns:
     triplet_loss
     xent_loss
  """
  # When fine-tuning a model, we do not restore the logits but instead we
  # randomly initialize the logits. The number of classes in the output of the
  # logit is the number of classes in specified Dataset.
  restore_logits = not FLAGS.fine_tune

  # Build inference Graph.
  logits1,logits2,endpoints = inception.inference(images, output_dims, for_training=True,
                               restore_logits=restore_logits,
                               scope=scope)

  # Build the portion of the Graph calculating the losses. Note that we will
  # assemble the total_loss using a custom function below.
  split_batch_size = images.get_shape().as_list()[0]
  triplet_loss = inception.triplet_loss(logits1,1.0)
  #TODO add a collection for triplet_loss
  tf.add_to_collection('triplet_loss',triplet_loss)
  inception.xent_loss(logits2, labels, batch_size=split_batch_size)

  # Assemble all of the losses for the current tower only.
  xent_losses = tf.get_collection(slim.losses.LOSSES_COLLECTION, scope)

  # Calculate the total loss for the current tower.
  # gather l2_loss from tf.get_variable()
  regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

  total_xent_loss = tf.add_n(xent_losses + regularization_losses, name='total_xent_loss')
  total_triplet_loss = tf.add_n([triplet_loss]+regularization_losses,name='total_triplet_loss')

  return total_xent_loss,total_triplet_loss

def _add_loss_summmaries(prefix,total_loss):
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  #TODO what about triplet?
  if 'triplet' in prefix:losses = tf.get_collection('triplet_loss')
  else:losses = tf.get_collection(slim.losses.LOSSES_COLLECTION)
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summmary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on TensorBoard.
    loss_name = re.sub('%s_[0-9]*/' % inception.TOWER_NAME, '', l.op.name)
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.scalar_summary(prefix+loss_name +' (raw)', l)
    tf.scalar_summary(prefix+loss_name, loss_averages.average(l))
  return loss_averages_op
#  with tf.control_dependencies([loss_averages_op]):
#    total_loss = tf.identity(total_loss)
#  return total_loss

def train_util(prefix,total_loss,global_step,opt,lr):
    loss_averages_op = _add_loss_summaries(prefix,total_loss)
    with tf.control_dependencies([loss_averages_op]):
        grads = opt.compute_gradients(total_loss)
    # Apply the gradients to adjust the shared variables.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add a summaries for the input processing and global_step.
    summaries.extend(input_summaries)

    # Add a summary to track the learning rate.
    summaries.append(tf.scalar_summary('learning_rate', lr))

    # Add histograms for gradients.
    for grad, var in grads:
      if grad is not None:
        summaries.append(
            tf.histogram_summary(prefix+var.op.name + '/gradients', grad))
    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
      summaries.append(tf.histogram_summary(prefix+var.op.name, var))

    # Track the moving averages of all trainable variables.
    # Note that we maintain a "double-average" of the BatchNormalization
    # global statistics. This is more complicated then need be but we employ
    # this for backward-compatibility with our previous models.
    variable_averages = tf.train.ExponentialMovingAverage(
        inception.MOVING_AVERAGE_DECAY, global_step)

    # Another possiblility is to use tf.slim.get_variables().
    variables_to_average = (tf.trainable_variables() +
                            tf.moving_average_variables())
    variables_averages_op = variable_averages.apply(variables_to_average)

    # Group all updates to into a single train op.
    batchnorm_updates_op = tf.group(*batchnorm_updates)
    #https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/image/cifar10/cifar10.py#L372
    #differ ?
    train_op = tf.group(apply_gradient_op, variables_averages_op,
                        batchnorm_updates_op)
    return train_op

def train():
  """Train on dataset for a number of steps."""
  with tf.Graph().as_default(), tf.device('/cpu:0'):
    # Create a variable to count the number of train() calls. This equals the
    # number of batches processed * FLAGS.num_gpus.
    global_step = tf.get_variable(
        'global_step', [],
        initializer=tf.constant_initializer(0), trainable=False)

    images = tf.placeholder(tf.float32,[None,FLAGS.image_size,FLAGS.image_size,3],name='input')
    labels = tf.placeholder(tf.int64,[None],name='labels')

    # Calculate the learning rate schedule.
    #TODO how about deal with num_examples
    num_examples = 50000
    num_batches_per_epoch = (num_examples /
                             FLAGS.batch_size)
    decay_steps = int(num_batches_per_epoch * FLAGS.num_epochs_per_decay)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                    global_step,
                                    decay_steps,
                                    FLAGS.learning_rate_decay_factor,
                                    staircase=True)

    # Create an optimizer that performs gradient descent.
    opt = tf.train.RMSPropOptimizer(lr, RMSPROP_DECAY,
                                    momentum=RMSPROP_MOMENTUM,
                                    epsilon=RMSPROP_EPSILON)

    input_summaries = copy.copy(tf.get_collection(tf.GraphKeys.SUMMARIES))

    # Number of classes in the Dataset label set plus 1.
    # Label 0 is reserved for an (unused) background class.
    num_classes = dataset.num_classes() + 1
    
    # Force all Variables to reside on the CPU.
    with slim.arg_scope([slim.variables.variable], device='/cpu:0'):
        total_xent_loss,total_triplet_loss = _total_loss(images_splits[i], labels_splits[i], 
                [128,num_classes],scope)

    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
    batchnorm_updates = tf.get_collection(slim.ops.UPDATE_OPS_COLLECTION,
                                                scope)
    triplet_train_op = train_util('tripletloss_',total_triplet_loss,global_step,opt,lr)
    xent_train_op = train_util('xentloss_',total_triplet_loss,global_step,opt,lr)

    # Create a saver.
    saver = tf.train.Saver(tf.all_variables())

    # Build the summary operation from the last tower summaries.
    summary_op = tf.merge_summary(summaries)

    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()

    # Start running operations on the Graph. allow_soft_placement must be set to
    # True to build towers on GPU, as some of the ops do not have GPU
    # implementations.
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    sess = tf.Session(config=tf.ConfigProto(
        gpu_options = gpu_options,
        allow_soft_placement=True,
        log_device_placement=FLAGS.log_device_placement))
    sess.run(init)

    if FLAGS.pretrained_model_checkpoint_path:
      assert tf.gfile.Exists(FLAGS.pretrained_model_checkpoint_path)
      variables_to_restore = tf.get_collection(
          slim.variables.VARIABLES_TO_RESTORE)
      restorer = tf.train.Saver(variables_to_restore)
      restorer.restore(sess, FLAGS.pretrained_model_checkpoint_path)
      print('%s: Pre-trained model restored from %s' %
            (datetime.now(), FLAGS.pretrained_model_checkpoint_path))

    summary_writer = tf.train.SummaryWriter(
        FLAGS.train_dir,
        graph_def=sess.graph.as_graph_def(add_shapes=True))

    #TODO start train
    for step in xrange(FLAGS.max_steps):
      start_time = time.time()
      _, loss_value = sess.run([train_op, loss])
      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step % 10 == 0:
        examples_per_sec = FLAGS.batch_size / float(duration)
        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
        print(format_str % (datetime.now(), step, loss_value,
                            examples_per_sec, duration))

      if step % 100 == 0:
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)

      # Save the model checkpoint periodically.
      if step % 5000 == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
