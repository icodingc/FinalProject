import time,os
import shutil
from datetime import datetime
from PIL import Image
from scipy import misc
import tensorflow as tf
import numpy as np
import vgg

tf.app.flags.DEFINE_integer("CONTENT_WEIGHT", 5e1, "Weight for content features loss")
tf.app.flags.DEFINE_integer("TV_WEIGHT", 1e-5, "Weight for total variation loss")
tf.app.flags.DEFINE_string("VGG_PATH", "/home/zhangxuesen/neuralstyle/imagenet-vgg-verydeep-19.mat",
        "Path to vgg model weights")

tf.app.flags.DEFINE_float("LEARNING_RATE", 10., "Learning rate")

tf.app.flags.DEFINE_string("CONTENT_IMAGE", "cat.jpg", "Content image to use")
tf.app.flags.DEFINE_string("CONTENT_LAYERS", "hybrid_pool4","Which VGG layer to extract")

tf.app.flags.DEFINE_boolean("RANDOM_INIT", True, "Start from random noise")
tf.app.flags.DEFINE_integer("NUM_ITERATIONS", 5000, "Number of iterations")

FLAGS = tf.app.flags.FLAGS

def total_variation_loss(layer):
    shape = tf.shape(layer)
    height = shape[1]
    width = shape[2]
    y = tf.slice(layer, [0,0,0,0], tf.pack([-1,height-1,-1,-1])) - tf.slice(layer, [0,1,0,0], [-1,-1,-1,-1])
    x = tf.slice(layer, [0,0,0,0], tf.pack([-1,-1,width-1,-1])) - tf.slice(layer, [0,0,1,0], [-1,-1,-1,-1])
    return tf.nn.l2_loss(x) / tf.to_float(tf.size(x)) + tf.nn.l2_loss(y) / tf.to_float(tf.size(y))

def get_content_features(content_path, content_layers):
    with tf.Graph().as_default() as g:
        image = tf.expand_dims(tf.convert_to_tensor(misc.imread(content_path),tf.float32), 0)
        net, _ = vgg.net(FLAGS.VGG_PATH, image)
        layers = []
        for layer in content_layers:
            layers.append(net[layer])

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction=0.4
        with tf.Session(config=config) as sess:
            return sess.run(layers+[image])

def main(argv=None):
    content_path = FLAGS.CONTENT_IMAGE
    content_layers = FLAGS.CONTENT_LAYERS.split(',')

    feat_and_image= get_content_features(content_path, content_layers)
    content_features=feat_and_image[:-1]
    image_t=feat_and_image[-1]

    image = tf.constant(image_t)
    random = tf.random_normal(image_t.shape)
    initial = tf.Variable(random if FLAGS.RANDOM_INIT else image)

    net, _ = vgg.net(FLAGS.VGG_PATH, initial)

    content_loss = 0
    for layer,feat in zip(content_layers,content_features):
        layer_size = tf.size(feat)
        content_loss += tf.nn.l2_loss(net[layer] - feat) / tf.to_float(layer_size)
    content_loss = FLAGS.CONTENT_WEIGHT * content_loss / len(content_layers)

    tv_loss = FLAGS.TV_WEIGHT * total_variation_loss(initial)
    total_loss = content_loss + tv_loss
    tf.scalar_summary('total_loss',total_loss)
    tf.image_summary('genpic',initial)

    global_step = tf.get_variable(
        'global_step', [],
        initializer=tf.constant_initializer(0), trainable=False)
    lr = tf.train.exponential_decay(FLAGS.LEARNING_RATE,
                                    global_step,
                                    3000,
                                    0.2,
                                    staircase=True)
    tf.scalar_summary('lr',lr)
    opt  = tf.train.MomentumOptimizer(FLAGS.LEARNING_RATE,0.9)
    train_op = opt.minimize(total_loss,global_step=global_step)

    output_image = tf.saturate_cast(tf.squeeze(initial), tf.uint8)

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction=0.4
    with tf.Session(config=config) as sess:

        merged = tf.merge_all_summaries()
        tmp_dir = 'tmp/log_'+FLAGS.CONTENT_LAYERS
        if os.path.exists(tmp_dir):shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir)
        writer = tf.train.SummaryWriter(tmp_dir,sess.graph)

        sess.run(tf.initialize_all_variables())

        for step in xrange(FLAGS.NUM_ITERATIONS):
            summary,_, loss_t = sess.run([merged,train_op, total_loss])
            writer.add_summary(summary,step)
            if step%10==0:
                print('{} step {} with loss {}'.format(datetime.now(),step, loss_t))
            if step>1500 and step%500==0:
                image_t = sess.run(output_image)
                misc.imsave('genimg/'+FLAGS.CONTENT_LAYERS+'_'+str(step)+'.jpg',
                        np.squeeze(image_t))

if __name__ == '__main__':
    tf.app.run()
