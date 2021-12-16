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
    """This function normalize the data to the range between 0 to 255
    """
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
    
class AxialAttention(layers.Layer):
    """spectral norm axial attention layer. 
    """
    def __init__(self, att_axis, groups, in_planes, out_planes, kernel_size, att_name, strides = 1):
        super(AxialAttention, self).__init__()
        
        self.att_axis = att_axis
        self.groups = groups
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.group_planes = out_planes // groups
        self.kernel_size = kernel_size
        self.strides = strides
        
        self.bn_qkv = BatchNormalization()
        self.bn_similarity = BatchNormalization()
        self.bn_output = BatchNormalization()
        
        self.qkv_transform = tfa.layers.SpectralNormalization(Conv1D(filters = 2 * out_planes, kernel_size = 1, strides = 1, padding = 'same', use_bias = False, data_format = "channels_first",
                                    kernel_initializer = tf.keras.initializers.RandomNormal(0.0, tf.math.sqrt(1.0 / self.in_planes))))
        
        self.relative = tf.Variable(tf.random.normal(shape = (self.group_planes * 2, kernel_size * 2 - 1), mean = 0.0, stddev = tf.math.sqrt(1.0 / self.group_planes)), 
                                    trainable = True, name = 'rel_' + att_name)
        query_index = tf.expand_dims(tf.range(kernel_size), axis = 0)
        key_index = tf.expand_dims(tf.range(kernel_size), axis = 1)
        relative_index = key_index - query_index + kernel_size - 1
        self.flatten_index = tf.reshape(relative_index, -1)
        
        if self.strides > 1:
            self.pooling = AveragePooling3D(pool_size = (strides, strides, strides), strides = strides)
            
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'att_axis': self.att_axis,
            'kernel': self.kernel_size,
            'groups': self.groups,
            'in_planes': self.in_planes,
            'out_planes': self.out_planes,
        })
        return config
            
    def call(self, x):
        # x: B x H x W x L x C
        if self.att_axis == 'H':
            x = tf.transpose(x, [0, 2, 3, 4, 1])
            
        elif self.att_axis == 'W':
            x = tf.transpose(x, [0, 1, 3, 4, 2])
            
        elif self.att_axis == 'L':
            x = tf.transpose(x, [0, 1, 2, 4, 3])
            
        # Let self.att_axis == 'H', then x: B x W x L x C x H
        B, W, L, C, H = x.shape
        x = tf.reshape(x, [B * W * L, C, H])
        
        qkv = self.bn_qkv(self.qkv_transform(x))
        q, k, v = tf.split(tf.reshape(qkv, [B * W * L, self.groups, self.group_planes * 2, H]), 
                           [self.group_planes // 2, self.group_planes // 2, self.group_planes], axis = -2)
    
        all_embeddings = tf.reshape(tf.gather(self.relative, self.flatten_index, axis = 1), [self.group_planes * 2, self.kernel_size, self.kernel_size])
        q_embedding, k_embedding, v_embedding = tf.split(all_embeddings, [self.group_planes // 2, self.group_planes // 2, self.group_planes], axis = 0)
        
        # q: BWL x Nh x dqh x H, q_embedding: dqh x H x H -> qr: BWL x Nh x H x H.
        qr = tf.einsum('bgci,cij -> bgij', q, q_embedding)
        kr = tf.transpose(tf.einsum('bgci,cij -> bgij', k, k_embedding), [0, 1, 3, 2])
        qk = tf.einsum('bgci, bgcj -> bgij', qr, kr)
        
        stacked_similarity = tf.concat([qk, qr, kr], axis = 1)
        stacked_similarity = tf.math.reduce_sum(tf.reshape(self.bn_similarity(stacked_similarity), [B * W * L, 3, self.groups, H, H]), axis = 1)
        
        similarity = tf.nn.softmax(stacked_similarity, axis = 3)
        sv = tf.einsum('bgij,bgcj -> bgci', similarity, v)
        sve = tf.einsum('bgij,cij -> bgci', similarity, v_embedding)
        stacked_output = tf.reshape(tf.concat([sv, sve], axis = -1), [B * W * L, self.out_planes * 2, H])
        output = tf.math.reduce_sum(tf.reshape(self.bn_output(stacked_output), [B, W, L, self.out_planes, 2, H]), axis = -2)
        
        if self.att_axis == 'H':
            output = tf.transpose(output, [0, 4, 1, 2, 3])
            
        elif self.att_axis == 'W':
            output = tf.transpose(output, [0, 1, 4, 2, 3])
            
        elif self.att_axis == 'L':
            output = tf.transpose(output, [0, 1, 2, 4, 3])
        
        if self.strides > 1:
            output = self.pooling(output)
            
        return output
    
class DBlock3D_att(layers.Layer):
    """downsampling blcok with spectral norm axial attention layer. 
    """
    def __init__(self,
                 filters,
                 downsample=True,
                 name='block',
                 groups=4):
        super(DBlock3D_att, self).__init__(name=name)
        self.filters = filters
        self.downsample = downsample
        self.groups = groups

        self.conv1 = SNConv3d(filters, (1, 1, 1), (1, 1, 1), use_bias=False, name='snconv_1')
        self.conv2 = SNConv3d(2*filters, (1, 1, 1), (2, 2, 1), use_bias=False, name='snconv_2')
        self.conv3 = SNConv3d(2*filters, (1, 1, 1), (2, 2, 1), use_bias=False, name='snconv_3')
        
        self.bn1 = BatchNormalization()
        self.bn2 = BatchNormalization()
        self.bn3 = BatchNormalization()
        
    def build(self, input_shape):
        self.height_block = AxialAttention(att_axis = 'H', groups = self.groups, in_planes = self.filters, out_planes = self.filters, kernel_size = int(input_shape[-4]), att_name = 'block' + '_h')
        self.width_block = AxialAttention(att_axis = 'W', groups = self.groups, in_planes = self.filters, out_planes = self.filters, kernel_size = int(input_shape[-3]), att_name = 'block' + '_w')
        self.layer_block = AxialAttention(att_axis = 'L', groups = self.groups, in_planes = self.filters, out_planes = self.filters, kernel_size = int(input_shape[-2]), att_name = 'block' + '_l',
                                          strides = 1)
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filters': self.filters,
            'downsample': self.downsample,
            'groups': self.groups,
        })
        return config

    def call(self, x, training=None):
        
        fx = self.conv1(x, training=training)
        
        fx = tf.nn.relu(self.bn1(fx))
        
        fx = self.height_block(fx)
        fx = self.width_block(fx)
        fx = self.layer_block(fx)
        fx = tf.nn.relu(fx)
        
        fx = self.conv2(fx, training=training)
        fx = self.bn2(fx)
        
        if self.downsample:
            x = self.conv3(x, training=training)
            x = self.bn3(x)
            
        return tf.nn.relu(x + fx)   
    
def d_hinge_loss(logits_real, logits_fake):
    """hinge loss for discriminator
    """
    loss_real = tf.nn.relu(1.0 - logits_real)
    loss_fake = tf.nn.relu(1.0 + logits_fake)
    return (tf.reduce_mean(loss_real) + tf.reduce_mean(loss_fake)) * (1.0 / batch) 

def g_hinge_loss(logits_fake):
    """hinge loss for generator
    """
    return -tf.reduce_mean(logits_fake) * (1.0 / batch)

def create_generator_att(gf_dim = 64, batch_size=5):
    """generator model with axial attention
    """
    input_layers = tf.keras.layers.Input(shape=(16, 16, 8), batch_size=batch_size) 
    input_layers0 = tf.expand_dims(input_layers, axis=-1)

    x11 = DBlock3D(gf_dim, downsample=True, apply_batchnorm=False, name='down_block_0')(input_layers0)
    x22 = DBlock3D_att(gf_dim*2, downsample=True, name='down_block_1')(x11)
    x33 = DBlock3D_att(gf_dim*4, downsample=True, name='down_block_2')(x22)
    x44 = DBlock3D_att(gf_dim*8, downsample=True, name='down_block_3')(x33)

    x1 = GBlock3D(gf_dim*8, upsample=True, name='up_block_0')(x44)
    x1 = Dropout(0.5)(x1)
    x1 = Concatenate()([x1, x33])
    x2 = GBlock3D(gf_dim*4, upsample=True, name='up_block_1')(x1)
    x2 = Dropout(0.5)(x2)
    x2 = Concatenate()([x2, x22])
    x3 = GBlock3D(gf_dim*2, upsample=True, name='up_block_2')(x2)

    x4 = Concatenate()([x3, x11])
    x = GBlock3D(gf_dim, upsample=True, name='up_block_' + str(4))(x4)
    x = tf.keras.layers.BatchNormalization(momentum=0.9999,
                                                epsilon=1e-5,
                                                name='up_bn_out')(x)

    x = tf.keras.layers.ReLU()(x)
    x = Dropout(0.25)(x)
    x = SNConv3d(1, (3, 3, 3), (1, 1, 1), name='up_conv_out')(x)
    x = tf.nn.tanh(x)
    x = x[:, :, :, :, 0]

    return tf.keras.models.Model(input_layers, x)

def create_generator(gf_dim = 64, batch_size=5):
    """generator model with convolution
    """
    input_layers = tf.keras.layers.Input(shape=(16, 16, 8), batch_size=batch_size) 
    input_layers0 = tf.expand_dims(input_layers, axis=-1)

    x11 = DBlock3D(gf_dim, downsample=True, name='down_block_0')(input_layers0)
    x22 = DBlock3D(gf_dim*2, downsample=True, name='down_block_1')(x11)
    x22 = BatchNormalization()(x22)
    x22 = tf.nn.relu(x22)
    x33 = DBlock3D(gf_dim*4, downsample=True, name='down_block_2')(x22)
    x33 = BatchNormalization()(x33)
    x33 = tf.nn.relu(x33)
    x44 = DBlock3D(gf_dim*8, downsample=True, name='down_block_3')(x33)
    x44 = BatchNormalization()(x44)
    x44 = tf.nn.relu(x44)

    x1 = GBlock3D(gf_dim*8, upsample=True, name='up_block_0')(x44)
    x1 = Dropout(0.5)(x1)
    x1 = Concatenate()([x1, x33])
    x2 = GBlock3D(gf_dim*4, upsample=True, name='up_block_1')(x1)
    x2 = Dropout(0.5)(x2)
    x2 = Concatenate()([x2, x22])
    x3 = GBlock3D(gf_dim*2, upsample=True, name='up_block_2')(x2)

    x4 = Concatenate()([x3, x11])
    x = GBlock3D(gf_dim, upsample=True, name='up_block_' + str(4))(x4)
    x = tf.keras.layers.BatchNormalization(momentum=0.9999,
                                                epsilon=1e-5,
                                                name='up_bn_out')(x)
    x = tf.keras.layers.ReLU()(x)
    x = Dropout(0.25)(x)
    x = SNConv3d(1, (3, 3, 3), (1, 1, 1), name='up_conv_out')(x)
    x = tf.nn.tanh(x)
    x = x[:, :, :, :, 0]

    return tf.keras.models.Model(input_layers, x)

def make_discriminator_model(image_size=16, filters=16, df_dim=32, batch_size=5):
    """discriminator model
    """
    input_layers = tf.keras.layers.Input((image_size, image_size, 8), batch_size=batch_size)
    input_layers0 = tf.expand_dims(input_layers, axis=-1)

    x = DBlock3D(df_dim, downsample=True, name='block_1')(input_layers0)
    x = Dropout(0.25)(x)
    x = DBlock3D(df_dim*2, downsample=True, name='block_2')(x)
    x = Dropout(0.25)(x)
    x = DBlock3D(df_dim*4, downsample=True, name='block_3')(x)
    x = Dropout(0.25)(x)
    x = DBlock3D(df_dim*8, downsample=False, name='block_4')(x)
    x = Dropout(0.25)(x)
    x = DBlock3D(df_dim*16, downsample=False, name='block_5')(x)
    x = tf.nn.relu(x)
    x = tf.reduce_sum(x, axis=[1, 2, 3])
    x = SNLinear(1, name='linear_out')(x)

    return tf.keras.models.Model([input_layers], x)
