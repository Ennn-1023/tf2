from tensorflow import keras
import tensorflow as tf


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
    