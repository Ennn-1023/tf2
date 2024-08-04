from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers


def gen_conv(x, cnum, ksize, stride=1, rate=1, name='conv',
             padding='SAME', activation=tf.nn.elu, training=True, dtype=tf.float32):
    x = conv2d(x, cnum, ksize, stride, dilation_rate=rate,
        activation=activation, padding=padding, name=name, dtype=dtype)
    return x

def gen_deconv(x, cnum, name='upsample', padding='SAME', training=True, dtype=tf.float32):
    x = tf.image.resize(x, size=[x.shape[1] * 2, x.shape[2] * 2], method='bilinear')
    x = gen_conv(x, cnum, 3, 1, name=name+'_conv', padding=padding,
            training=training, dtype=dtype)
    return x
def conv2d(x, output_dim, ksize, stride, dilation_rate=1, activation=None, padding='same', name='conv',
           dtype=tf.float32):
    # 使用 Keras 的 Conv2D 層來替代自定義的卷積操作
    conv_layer = tf.keras.layers.Conv2D(
        filters=output_dim,
        kernel_size=ksize,
        strides=stride,
        dilation_rate=dilation_rate,
        activation=activation,  # 激活函數
        padding=padding,
        name=name,
        dtype=dtype,
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.05),  # 權重初始化
        bias_initializer=tf.keras.initializers.Constant(0.0)  # 偏置初始化
    )

    # 將輸入 x 通過 Conv2D 層進行卷積操作
    conv = conv_layer(x)
    return conv

def gen_conv_gated(x, cnum, ksize, stride=1, rate=1, name='conv',
             padding='SAME', slim=True, activation=None, training=True, dtype=tf.float32):
    x1 = conv2d(x, cnum, ksize, stride, dilation_rate=rate,
        activation=None, padding=padding, name=name+'_feat', dtype=dtype)
    x2 = conv2d(x, cnum, ksize, stride, dilation_rate=rate,
        activation=None, padding=padding, name=name+'_gate', dtype=dtype)
    #x1, x2 = tf.split(x, 2, axis=3)
    x = tf.sigmoid(x2) * tf.nn.elu(x1)
    return x

def conv2d_ds(x, output_dim, ksize, stride, dilation_rate=1, activation=None, \
                        padding='SAME', name='conv', dtype=tf.float32):
    # 使用 Keras 的 SeparableConv2D 進行深度可分離卷積
    conv_layer = layers.SeparableConv2D(
        filters=output_dim,
        kernel_size=ksize,
        strides=stride,
        dilation_rate=dilation_rate,
        padding=padding,
        activation=activation,
        depthwise_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.05),
        pointwise_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.05),
        bias_initializer=tf.keras.initializers.Constant(0.0),
        name=name,
        dtype=dtype
    )

    # 將輸入 x 通過 SeparableConv2D 層進行卷積操作
    y = conv_layer(x)
    return y

def gen_deconv_gated_ds(x, cnum, name='upsample', padding='SAME', training=True, dtype=tf.float32):
    x = tf.image.resize(x, size=[x.shape[1] * 2, x.shape[2] * 2], method='bilinear')
    #x = resize(x, func=tf.compat.v1.image.resize_bilinear)
    x = gen_conv_gated_ds( x, cnum, 3, 1, name=name+'_conv', padding=padding,
        training=training, dtype=dtype)
    return x

def gen_conv_gated_ds(x, cnum, ksize, stride=1, rate=1, name='conv',
             padding='SAME', slim=True, activation=None, training=True, dtype=tf.float32):
    x1 = conv2d(x, cnum, ksize, stride, dilation_rate=rate,
        activation=None, padding=padding, name=name, dtype=dtype)
    x2 = conv2d_ds(x, cnum, 3, stride, dilation_rate=1,
        activation=None, padding=padding, name=name, dtype=dtype)
    x = tf.sigmoid(x2) * tf.nn.elu(x1)
    return x

def gen_conv_gated_slice(x, cnum, ksize, stride=1, rate=1, name='conv',
             padding='SAME', slim=True, activation=None, training=True, dtype=tf.float32):
    x1 = conv2d(x, cnum, ksize, stride, dilation_rate=rate,
        activation=None, padding=padding, name=name+'_feat', dtype=dtype)
    x2 = conv2d(x, 1, ksize, stride, dilation_rate=rate,
        activation=None, padding=padding, name=name+'_gate', dtype=dtype)
    #x1, x2 = tf.split(x, [cnum,1], axis=3)
    x = tf.sigmoid(x2) * tf.nn.elu(x1)
    return x

def gen_deconv_gated_slice(x, cnum, name='upsample', padding='SAME', training=True, dtype=tf.float32):
    #with tf.compat.v1.variable_scope(name):
        #x = resize(x, func=tf.compat.v1.image.resize_bilinear)
    x = tf.image.resize(x, size=[x.shape[1] * 2, x.shape[2] * 2], method='bilinear')
    x = gen_conv_gated_slice(  x, cnum, 3, 1, name=name+'_conv', padding=padding,
        training=training, dtype=dtype)
    return x

def gen_deconv_gated(x, cnum, name='upsample', padding='SAME', training=True, dtype=tf.float32):
    x = tf.image.resize(x, size=[x.shape[1] * 2, x.shape[2] * 2], method='bilinear')
    #with tf.variable_scope(name):
    #   x = resize(x, func=tf.image.resize_bilinear)
    x = gen_conv_gated( x, cnum, 3, 1, name=name+'_conv', padding=padding,
        training=training, activation=None, dtype=dtype)
    return x

def dis_conv(x, cnum, ksize=5, stride=2, name='conv', dtype=tf.float32):
    x = conv2d(x, cnum, ksize, stride, padding='SAME', name=name, dtype=dtype,activation='leaky_relu')
    # x = tf.nn.leaky_relu(x)
    return x

def flatten(x, name=""):
    return tf.reshape(x, [x.get_shape().as_list()[0], -1], name=name)

class Discriminator_block(keras.layers.Layer):
    def __init__(self, name, training=True, nc=64, ksize=5, stride=2, dtype=tf.float32):
        super(Discriminator_block, self).__init__(name=name)
        self.training = training
        self.nc = nc
        self.ksize = ksize
        self.stride = stride
        self.Reshape = keras.layers.Reshape()
        
    def call(self, x):
        stride = self.stride
        nc = self.nc
        dtype = self.dtype
        ksize = self.ksize
        x = conv2d(x, nc, ksize, stride, padding='SAME', name='conv1', dtype=dtype)
        x = keras.layers.LeakyReLU()(x)
        x = conv2d(x, nc*2, ksize, stride, padding='SAME', name='conv2', dtype=dtype)
        x = keras.layers.LeakyReLU()(x)
        x = conv2d(x, nc*4, ksize, stride, padding='SAME', name='conv3', dtype=dtype)
        x = keras.layers.LeakyReLU()(x)
        x = conv2d(x, nc*4, ksize, stride, padding='SAME', name='conv4', dtype=dtype)
        x = keras.layers.LeakyReLU()(x)
        x = conv2d(x, nc*4, ksize, stride, padding='SAME', name='conv5', dtype=dtype)
        x = keras.layers.LeakyReLU()(x)
        x = conv2d(x, nc*4, ksize, stride, padding='SAME', name='conv6', dtype=dtype)
        x = flatten(x)
        return x
    