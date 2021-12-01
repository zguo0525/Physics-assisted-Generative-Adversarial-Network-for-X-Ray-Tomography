# tensorflow model used in "Physics-assisted Generative Adversarial Network"
# written and maintained by Zhen Guo
# =============================================================================

from tensorflow.keras.layers import *
import os
import time
from IPython import display
import math
from tqdm import tqdm
from scipy.io import loadmat
import numpy as np
from tensorflow.keras import optimizers, regularizers
import tensorflow as tf
from tensorflow.keras import backend as K
import tensorflow_addons as tfa
from tensorflow.keras import layers

def norm_to_255(tensor):
    "This function normalize the data to the range between 0 to 255
    "
    tf_max = tf.math.reduce_max(tensor)
    tf_min = tf.math.reduce_min(tensor)
    return 255 * (tensor - tf_min) / (tf_max - tf_min)

def npcc(truth, guess):
    """Compare a complexed guess image and a true image using npcc
    """
    guess_real = guess
    truth_real = truth
    
    fsp_real = guess_real - tf.reduce_mean(guess_real)
    fst_real = truth_real - tf.reduce_mean(truth_real)

    devP_real = tf.math.reduce_std(guess_real)
    devT_real = tf.math.reduce_std(truth_real)
    
    loss_pcc_real = (-1) * tf.reduce_mean(fsp_real * fst_real) / K.clip(devP_real * devT_real , K.epsilon(), None)    

    return loss_pcc_real

def _l2normalize(v, eps=1e-12):
    """l2normalize
    """
    return v / (tf.reduce_sum(v**2)**0.5 + eps)


def spectral_norm(weights, u, num_iters, training):
    """spectral_norm
    """
    w_shape = weights.shape.as_list()
    w_mat = tf.reshape(weights, [-1, w_shape[-1]])

    u_ = u
    for _ in range(num_iters):
        v_ = _l2normalize(tf.matmul(u_, w_mat, transpose_b=True))
        u_ = _l2normalize(tf.matmul(v_, w_mat))

    sigma = tf.squeeze(tf.matmul(tf.matmul(v_, w_mat), u_, transpose_b=True))
    w_mat /= sigma
    w_bar = tf.reshape(w_mat, w_shape)
    if training:
        u.assign(u_)
    return w_bar

class SNLinear(layers.Layer):
    """spectral norm linear layer
    """
    def __init__(self,
                 units,
                 use_bias=True,
                 sn_iters=1,
                 initializer=tf.initializers.orthogonal(),
                 name='snlinear'):
        super(SNLinear, self).__init__(name=name)
        self.units = units
        self.use_bias = use_bias
        self.sn_iters = sn_iters
        self.initializer = initializer

    def build(self, input_shape):
        in_features = int(input_shape[-1])
        kernel_shape = [in_features, self.units]
        self.kernel = self.add_weight('kernel', shape=kernel_shape,
                                      initializer=self.initializer)
        if self.use_bias:
            self.bias = self.add_weight('bias', shape=[self.units],
                                        initializer=tf.zeros_initializer())
        self.u = self.add_weight('u', shape=[1, self.units], trainable=False)

    def call(self, x, training=None):
        w_bar = spectral_norm(self.kernel, self.u, self.sn_iters, training)
        x = tf.matmul(x, w_bar)
        if self.use_bias:
            return x + self.bias
        else:
            return x
    
class SNConv3d(layers.Layer):
    """spectral norm Conv3D layer
    """
    def __init__(self,
                 filters,
                 kernel_size=(3, 3, 3),
                 strides=(2, 2, 2),
                 sn_iters=1,
                 initializer=tf.initializers.orthogonal(),
                 name='snconv2d'):
        super(SNConv3d, self).__init__(name=name)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.sn_iters = sn_iters
        self.initializer = initializer
        self.conv3d = tfa.layers.SpectralNormalization(tf.keras.layers.Conv3D(self.filters,
                                           self.kernel_size,
                                           self.strides,
                                           padding='SAME',
                                            use_bias=True,
                                           kernel_initializer=self.initializer,
                                           bias_initializer='zeros'))
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'sn_iters': self.sn_iters,
            'initializer': self.initializer,
        })
        return config

    def call(self, x, training=None):
        x = self.conv3d(x)
        return x

def usample3d(x):
    """upsample in h and w of 3D data input. 
    """
    b, h, w, c, ch = x.shape.as_list()
    transposed = tf.transpose(x, [0,3,1,2,4])
    reshaped = tf.reshape(transposed, [b*c,h,w,ch])

    # and finally we use tf.image.resize_images
    new_size = tf.constant([h*2, w*2])
    resized = tf.image.resize(reshaped, new_size, method='nearest')

    undo_reshape = tf.reshape(resized, [b,c,h*2,w*2,ch])

    undo_transpose = tf.transpose(undo_reshape, [0,2,3,1,4]) 
    return undo_transpose


class GBlock3D(layers.Layer):
    """upsampling blcok with spectral norm Conv3D layer. 
    """
    def __init__(self,
                 filters,
                 upsample=True,
                 name='block'):
        super(GBlock3D, self).__init__(name=name)
        self.filters = filters
        self.upsample = upsample
        self.bn1 = BatchNormalization()
        self.bn2 = BatchNormalization()
        self.conv1 = SNConv3d(filters, (3, 3, 3), (1, 1, 1), name='snconv_1')
        self.conv2 = SNConv3d(filters, (3, 3, 3), (1, 1, 1), name='snconv_2')
        self.conv3 = SNConv3d(filters, (1, 1, 1), (1, 1, 1), name='snconv_3')
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filters': self.filters,
            'upsample': self.upsample,
        })
        return config

    def call(self, inputs, training=None):
        x = inputs
        fx = tf.nn.relu(self.bn1(x))
        if self.upsample:
            fx = usample3d(fx)
        fx = self.conv1(fx, training=training)
        fx = tf.nn.relu(self.bn2(fx))
        fx = self.conv2(fx, training=training)
        if self.upsample:
            x = usample3d(x)
        x = self.conv3(x, training=training)

        return x + fx
    
def dsample3d(x):
    """downsample in h and w of 3D data input. 
    """
    x = tf.nn.avg_pool(x, [1, 2, 2, 1, 1], [1, 2, 2, 1, 1], 'VALID')
    return x
    
class DBlock3D(layers.Layer):
    """downsampling blcok with spectral norm Conv3D layer. 
    """
    def __init__(self,
                 filters,
                 downsample=True,
                 name='block'):
        super(DBlock3D, self).__init__(name=name)
        self.filters = filters
        self.downsample = downsample

        self.conv1 = SNConv3d(filters, (3, 3, 3), (1, 1, 1), name='snconv_1')
        self.conv2 = SNConv3d(filters, (3, 3, 3), (1, 1, 1), name='snconv_2')
        self.conv3 = SNConv3d(filters, (1, 1, 1), (1, 1, 1), name='snconv_3')
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filters': self.filters,
            'downsample': self.downsample,
        })
        return config

    def call(self, x, training=None):
        fx = tf.nn.relu(x)
        fx = self.conv1(fx, training=training)
        fx = tf.nn.relu(fx)
        fx = self.conv2(fx, training=training)
        x = self.conv3(x, training=training)
        
        if self.downsample:
            fx = dsample3d(fx)
            x = dsample3d(x)
        return x + fx
