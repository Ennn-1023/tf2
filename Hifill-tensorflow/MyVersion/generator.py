import tensorflow as tf
from tensorflow import keras
import numpy as np

# generator gated conv2d layer
class gen_conv_gated_ds(keras.layers.Layer): # done
    def __init__(self, output_dim, kernel_size, strides=1, rate=1, padding='SAME'
                 , activation=None, trainable=True, name='gated_conv', dtype=tf.float32, **kwargs):
        super(gen_conv_gated_ds, self).__init__(trainable, name, dtype, **kwargs)
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.strides = strides
        self.rate = rate
        self.padding = padding
        self.activation = activation
        self.trainable = trainable
        self.name = name
        self.dtype = dtype

    def call(self, input):
        # x1 = conv2d
        # x2 = conv2d_ds
        x1 = keras.layers.Conv2D(self.output_dim, self.kernel_size, self.strides, dilation_rate=self.rate, 
                                 padding=self.padding, activation=self.activation)(input)
        x1 = keras.layers.ELU()(x1)

        x2 = keras.layers.SeparableConv2D(filters=self.output_dim, kernel_size=self.kernel_size, strides=self.strides
                                          , padding=self.padding, activation=self.activation)(input)
        
        x2 = keras.layers.Softmax()(x2)
        output = keras.layers.multiply()([x1, x2])
        return output
class gen_deconv_gated_ds(keras.layers.Layer):
    def __init__(self, output_dim, kernel_size, strides=1, rate=1, padding='SAME'
                 , activation=None, trainable=True, name='gated_deconv', dtype=tf.float32, **kwargs):
        super(gen_deconv_gated_ds, self).__init__(trainable, name, dtype, **kwargs)
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.strides = strides
        self.rate = rate
        self.padding = padding
        self.activation = activation
        self.trainable = trainable
        self.name = name
        self.dtype = dtype

# define the generator model
class generator(keras.models.Model):
    def __init__(self, input_shape=[512, 512], config=None):
        super().__init__()
        self.input_shape = [input_shape.copy().append(3), input_shape.copy().append(1)]
        self.config = config

    def call(self, input):
        img_input = tf.keras.layers.Input(shape=self.input_shape[0], batch_size=self.config.BATCH_SIZE)
        mask_input = tf.keras.layers.Input(shape=self.input_shape[1], batch_size=self.config.BATCH_SIZE)
        xnow = keras.layers.concatenate([img_input, mask_input], axis=3) # (512, 512, 4)
        activations = [img_input]
        # encoder
        sz_t = self.input_shape[0][0]
        x = xnow
        channelNum = self.GEN_NC
        channelNum = max(4, channelNum // (sz_t // 512)) // 2
        while sz_t > self.config.BOTTLENECK_SIZE:
            channelNum *= 2
            sz_t //= 2
            # kkernal = 5 if sz_t == self.input_shape[0][0] else 3
            # 檢查一下這行原本有沒有用到
            x = gen_conv_gated_ds(output_dim=channelNum, kernel_size=3, strides=2, rate=1, padding='SAME', name="en_down_"+str(sz_t))(x)
            shortCut = x
            x = gen_conv_gated_ds(output_dim=channelNum, kernel_size=3, strides=1, rate=1, padding='SAME', name="en_conv_"+str(sz_t))(x)
            # apply residual
            x = keras.layers.add([shortCut, x])
            activations.append(x)
        
        # dilated conv
        # x = dilate_block2(x, channelNum, 3, rate=1, name='re_en_dilated_1')


                


