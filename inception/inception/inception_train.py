from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
from datetime import datetime
import os.path
import sys
import re
import time

import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf

from inception import image_processing
from inception import inception_model as inception
from inception.slim import slim
from inception.util import tripletnet as util

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
tf.app.flags.DEFINE_float('alpha', 0.4,
                          """hard margin for triplet loss.""")

# Constants dictating the learning rate schedule.
RMSPROP_DECAY = 0.9                # Decay term for RMSProp.
RMSPROP_MOMENTUM = 0.9             # Momentum in RMSProp.
RMSPROP_EPSILON = 1.0              # Epsilon term for RMSProp.

def _total_loss(prefix, images, labels, output_dims, scope):
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

  # Calculate the total loss for the current tower.
  # gather l2_loss from tf.get_variable()
  regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

  if 'triplet' in prefix:
    triplet_loss = inception.triplet_loss(logits2,FLAGS.alpha)
    tf.add_to_collection('triplet_loss',triplet_loss)
    losses = tf.get_collection('triplet_loss',scope)
    total_loss = tf.add_n(losses + regularization_losses, name='total_triplet_loss')
  else:
    acc = inception.xent_loss(logits1, labels, batch_size=split_batch_size)
    losses = tf.get_collection(slim.losses.LOSSES_COLLECTION,scope)
    total_loss = tf.add_n(losses+regularization_losses,name='total_xent_loss')
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
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
  with tf.control_dependencies([loss_averages_op]):
    total_loss = tf.identity(total_loss)
  if 'triplet' not in prefix:return total_loss,endpoints['logits2'],acc
  return total_loss , endpoints['logits2']

def _average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(0, grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


def xent_train(dataset):
  """Train on dataset for a number of steps."""
  with tf.Graph().as_default(), tf.device('/cpu:0'):
    # Create a variable to count the number of train() calls. This equals the
    # number of batches processed * FLAGS.num_gpus.
    global_step = tf.get_variable(
        'global_step', [],
        initializer=tf.constant_initializer(0), trainable=False)

    # Calculate the learning rate schedule.
    num_batches_per_epoch = (dataset.num_examples_per_epoch() /
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

    # Get images and labels for ImageNet and split the batch across GPUs.
    assert FLAGS.batch_size % FLAGS.num_gpus == 0, (
        'Batch size must be divisible by number of GPUs')
    split_batch_size = int(FLAGS.batch_size / FLAGS.num_gpus)

    # Override the number of preprocessing threads to account for the increased
    # number of GPU towers.
    num_preprocess_threads = FLAGS.num_preprocess_threads * FLAGS.num_gpus
    images, labels = image_processing.distorted_inputs(
        dataset,
        num_preprocess_threads=num_preprocess_threads)

    input_summaries = copy.copy(tf.get_collection(tf.GraphKeys.SUMMARIES))

    # Number of classes in the Dataset label set plus 1.
    # Label 0 is reserved for an (unused) background class.
    #TODO care about this
    num_classes = dataset.num_classes() + 1
    
     # Split the batch of images and labels for towers.
    images_splits = tf.split(0, FLAGS.num_gpus, images)
    labels_splits = tf.split(0, FLAGS.num_gpus, labels)

    # Calculate the gradients for each model tower.
    tower_grads = []
    for i in xrange(FLAGS.num_gpus):
      with tf.device('/gpu:%d' % i):
        with tf.name_scope('%s_%d' % (inception.TOWER_NAME, i)) as scope:
          # Force all Variables to reside on the CPU.
          with slim.arg_scope([slim.variables.variable], device='/cpu:0'):
            # Calculate the loss for one tower of the ImageNet model. This
            # function constructs the entire ImageNet model but shares the
            # variables across all towers.
            loss,_,acc = _total_loss('xentloss',images_splits[i], labels_splits[i], [num_classes,256],
                               scope)

          # Reuse variables for the next tower.
          tf.get_variable_scope().reuse_variables()

          # Retain the summaries from the final tower.
          summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

          # Retain the Batch Normalization updates operations only from the
          # final tower. Ideally, we should grab the updates from all towers
          # but these stats accumulate extremely fast so we can ignore the
          # other stats from the other towers without significant detriment.
          batchnorm_updates = tf.get_collection(slim.ops.UPDATE_OPS_COLLECTION,
                                                scope)

          # Calculate the gradients for the batch of data on this ImageNet
          # tower.
          grads = opt.compute_gradients(loss)

          # Keep track of the gradients across all towers.
          tower_grads.append(grads)

    # We must calculate the mean of each gradient. Note that this is the
    # synchronization point across all towers.
    #grads = _average_gradients(tower_grads)
    grads = tower_grads[0]

    # Add a summaries for the input processing and global_step.
    summaries.extend(input_summaries)

    # Add a summary to track the learning rate.
    summaries.append(tf.scalar_summary('learning_rate', lr))

    # Add histograms for gradients.
    for grad, var in grads:
      if grad is not None:
        summaries.append(
            tf.histogram_summary(var.op.name + '/gradients', grad))

    # Apply the gradients to adjust the shared variables.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
      summaries.append(tf.histogram_summary(var.op.name, var))

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
    train_op = tf.group(apply_gradient_op, variables_averages_op,
                        batchnorm_updates_op)

    # Create a saver.
    saver = tf.train.Saver(tf.all_variables())

    # Build the summary operation from the last tower summaries.
    summary_op = tf.merge_summary(summaries)

    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()

    # Start running operations on the Graph. allow_soft_placement must be set to
    # True to build towers on GPU, as some of the ops do not have GPU
    # implementations.
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
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

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.train.SummaryWriter(
        FLAGS.train_dir,
        graph_def=sess.graph.as_graph_def(add_shapes=True))

    for step in xrange(FLAGS.max_steps):
      start_time = time.time()
      _, loss_value,acu = sess.run([train_op, loss,acc])
      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step % 10 == 0:
        examples_per_sec = FLAGS.batch_size / float(duration)
        format_str = ('%s: step %d, loss = %.2f, acc = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
        print(format_str % (datetime.now(), step, loss_value,acu,
                            examples_per_sec, duration))

      if step % 100 == 0:
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)

      # Save the model checkpoint periodically.
      if step % 5000 == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
#TODO queue batch_loader (blog)
def triplet_train(dataset):
  """Train on dataset for a number of steps."""
  with tf.Graph().as_default(), tf.device('/cpu:0'):
    # Create a variable to count the number of train() calls. This equals the
    # number of batches processed * FLAGS.num_gpus.
    global_step = tf.get_variable(
        'global_step', [],
        initializer=tf.constant_initializer(0), trainable=False)

    # Calculate the learning rate schedule.
    num_examples_per_epoch = 50000
    num_batches_per_epoch = (num_examples_per_epoch /
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

    # Get images and labels for ImageNet and split the batch across GPUs.
    assert FLAGS.batch_size % FLAGS.num_gpus == 0, (
        'Batch size must be divisible by number of GPUs')
    split_batch_size = int(FLAGS.batch_size / FLAGS.num_gpus)

    # Override the number of preprocessing threads to account for the increased
    # number of GPU towers.
    num_preprocess_threads = FLAGS.num_preprocess_threads * FLAGS.num_gpus
    #TODO get images,positive,negatives
    images = tf.placeholder(tf.float32,[None,32,32,3])
    labels = tf.placeholder(tf.int32,[None])

    # Number of classes in the Dataset label set plus 1.
    # Label 0 is reserved for an (unused) background class.
    #TODO care about this[equal to classification]
    num_classes = 11
    
     # Split the batch of images and labels for towers.
    images_splits = tf.split(0, FLAGS.num_gpus, images)
    labels_splits = tf.split(0, FLAGS.num_gpus, labels)

    # Calculate the gradients for each model tower.
    tower_grads = []
    for i in xrange(FLAGS.num_gpus):
      with tf.device('/gpu:%d' % i):
        with tf.name_scope('%s_%d' % (inception.TOWER_NAME, i)) as scope:
          # Force all Variables to reside on the CPU.
          with slim.arg_scope([slim.variables.variable], device='/cpu:0'):
            # Calculate the loss for one tower of the ImageNet model. This
            # function constructs the entire ImageNet model but shares the
            # variables across all towers.
            loss,embedding = _total_loss('tripletloss',images_splits[i], labels_splits[i], [num_classes,128],
                               scope)

          # Reuse variables for the next tower.
          tf.get_variable_scope().reuse_variables()

          # Retain the summaries from the final tower.
          #summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

          # Retain the Batch Normalization updates operations only from the
          # final tower. Ideally, we should grab the updates from all towers
          # but these stats accumulate extremely fast so we can ignore the
          # other stats from the other towers without significant detriment.
          batchnorm_updates = tf.get_collection(slim.ops.UPDATE_OPS_COLLECTION,
                                                scope)

          # Calculate the gradients for the batch of data on this ImageNet
          # tower.
          grads = opt.compute_gradients(loss)

          # Keep track of the gradients across all towers.
          tower_grads.append(grads)

    # We must calculate the mean of each gradient. Note that this is the
    # synchronization point across all towers.
#    grads = _average_gradients(tower_grads)
    grads = tower_grads[0]

    
    #tf.scalar_summary('loss', loss)
    #tf.scalar_summary('learning_rate', lr)

    # Add histograms for gradients.

    # Apply the gradients to adjust the shared variables.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)


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
    train_op = tf.group(apply_gradient_op, variables_averages_op,
                        batchnorm_updates_op)

    # Create a saver.
    saver = tf.train.Saver(tf.all_variables())

    # Build the summary operation from the last tower summaries.
    #summary_op = tf.merge_all_summaries()

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

#    summary_writer = tf.train.SummaryWriter(
#        FLAGS.train_dir,
#        graph_def=sess.graph.as_graph_def(add_shapes=True))
    step = 0
    while step < FLAGS.max_steps:
      start_time = time.time()
      #TODO get triplets and train
      print('Loading traing data')
      start = time.time()
      class_per_batch = 10
      images_per_class = 1000
      random_flip = False
      image_paths,num_per_class = util.sample_class(dataset,class_per_batch,images_per_class)
      image_data = util.load_data(image_paths,random_flip,)
      load_time = time.time() -start
      print('Loaded %d images in %.2f seconds' % (image_data.shape[0], load_time))

#      print('Selecting suitable triplets for training')
      start_time = time.time()
      emb_list = []
      #Run a forward pass for the sampled images
      nrof_examples_per_epoch = class_per_batch * images_per_class
      nrof_batches_per_epoch = int(np.floor(nrof_examples_per_epoch / FLAGS.batch_size))
      for i in xrange(0,nrof_examples_per_epoch,FLAGS.batch_size):
        batch_ = image_data[i:i+FLAGS.batch_size]
        feed_dict = {images: batch_, labels: np.zeros(batch_.shape[0])}
        emb_list += sess.run([embedding], feed_dict=feed_dict)
      # Stack the embeddings to a nrof_examples_per_epoch x 128 matrix
      emb_array = np.vstack(emb_list)  
      print('Forward %d images in %.2f seconds' % (emb_array.shape[0], time.time()-start_time))

      # Select triplets based on the embeddings
      start_time = time.time()
      triplets, nrof_random_negs, nrof_triplets = util.select_triplets(
            emb_array, num_per_class, image_data, class_per_batch, FLAGS.alpha)
      selection_time = time.time() - start_time
      print('(nrof_random_negs, nrof_triplets) = (%d, %d): time=%.3f seconds' % (
      nrof_random_negs, nrof_triplets, selection_time))
      
      # triplets (a,p,n)
      a,p,n = triplets
      print('Before aug.',a.shape)
      anc = [a]
      pos = [p]
      neg = [n]
      for _ in xrange(3):
          a_,p_,n_ = shuffle(a,p,n)
          anc.append(a_)
          pos.append(p_)
          neg.append(n_)
      anc = np.vstack(anc)
      pos = np.vstack(pos)
      neg = np.vstack(neg)
      print('After aug.',anc.shape)

      # Perform training on the selected triplets   
      while step < FLAGS.max_steps:
        start = time.time()
        bs = int(FLAGS.batch_size/3)
        for i in xrange(0,nrof_triplets,bs):
          batch_a = anc[i:i+bs]
          batch_p = pos[i:i+bs]
          batch_n = neg[i:i+bs]
          batch_ = np.vstack([batch_a,batch_p,batch_n])
          feed_dict={images:batch_,labels:np.zeros(batch_.shape[0])}

          _, loss_value = sess.run([train_op, loss],feed_dict)
          duration = time.time() - start_time

          if step % 10 == 0:
            examples_per_sec = FLAGS.batch_size / float(duration)
            format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
            print(format_str % (datetime.now(), step, loss_value,
                            examples_per_sec, duration))

          if step % 5000 == 0 or (step + 1) == FLAGS.max_steps:
            checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)
          step += 1


