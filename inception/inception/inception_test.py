from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from inception.slim import inception_model as nin

x = np.random.randn(30,32,32,3)
x = tf.convert_to_tensor(x,tf.float32)
rst = nin.nin(x,[10,128])
