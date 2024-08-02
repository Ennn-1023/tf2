from tensorflow import keras
import tensorflow as tf
from ops import conv2d

def dis_conv(x, cnum, ksize=5, stride=2, name='conv', training=True, dtype=tf.float32):
    x = conv2d(x, cnum, ksize, stride, padding='SAME', name=name, dtype=dtype,activation='leaky_relu')
    # x = tf.nn.leaky_relu(x)
    return x

def flatten(x, name=""):
    return tf.reshape(x, [x.get_shape().as_list()[0], -1], name=name)

class Discriminator_block(keras.layers.Layer):
    def __init__(self, name, training=True, nc=64):
        super(Discriminator_block, self).__init__(name=name)
        self.training = training
        self.nc = nc
    def call(self, x):
        x = dis_conv(x, nc, name='conv1', training=self.training)
        x = dis_conv(x, nc*2, name='conv2', training=self.training)
        x = dis_conv(x, nc*4, name='conv3', training=self.training)
        x = dis_conv(x, nc*4, name='conv4', training=self.training)
        x = dis_conv(x, nc*4, name='conv5', training=self.training)
        x = dis_conv(x, nc*4, name='conv6', training=self.training)
        x = tf.reshape(x, [x.get_shape().as_list()[0], -1], name='reshape')
        return x
    