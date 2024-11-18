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
                                 padding=self.padding, activation=self.activation, dtype=self.dtype,
                                 kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.05),
                                 bias_initializer=keras.initializers.Constant(0.0))(input)
        x1 = keras.layers.ELU()(x1)
        x2 = keras.layers.SeparableConv2D(filters=self.output_dim, kernel_size=self.kernel_size, strides=self.strides, 
                                          padding=self.padding, activation=self.activation, dtype = self.dtype,
                                          depthwise_initializer=keras.initializers.TruncatedNormal(stddev=0.05),
                                          pointwise_initializer=keras.initializers.TruncatedNormal(stddev=0.05),
                                          bias_initializer=keras.initializers.Constant(0.0))(input)
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
    def call(self, input):
        raise NotImplementedError


class dilate_block2(keras.layers.Layer): # not checked yet
    def __init__(self, input, output_dim, kernel_size, rate, name='dilate_block2', **kwargs):
        super(dilate_block2, self).__init__(name=name, **kwargs)
        self.input = input
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.rate = rate
        self.name = name
    def call(self, input):
        channelNum = input.get_shape()[3]
        x = gen_conv_gated_ds(output_dim=channelNum, kernel_size=3, strides=1, rate=1, padding='SAME', name="dilated_1")(input)
        x = gen_conv_gated_ds(output_dim=channelNum, kernel_size=3, strides=1, rate=2, padding='SAME', name="dilated_2")(x)
        x = gen_conv_gated_ds(output_dim=channelNum, kernel_size=3, strides=1, rate=4, padding='SAME', name="dilated_4")(x)
        x = gen_conv_gated_ds(output_dim=channelNum, kernel_size=3, strides=1, rate=8, padding='SAME', name="dilated_8")(x)
        x = gen_conv_gated_ds(output_dim=channelNum, kernel_size=3, strides=1, rate=16, padding='SAME', name="dilated_16")(x)
        return x

# define the generator model
class generator(keras.models.Model):
    def __init__(self, input_shape=[512, 512], config=None):
        super().__init__()
        self.input_shape = [input_shape.copy().append(3), input_shape.copy().append(1)]
        self.config = config

    def call(self, input, training=True):
        img_input = keras.layers.Input(shape=(512, 512, 3), batch_size=config.BATCH_SIZE)
        mask_input = keras.layers.Input(shape=(512, 512, 1), batch_size=config.BATCH_SIZE)
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
            x = gen_conv_gated_ds(output_dim=channelNum, kernel_size=3, strides=2, rate=1, padding='SAME', name="encode_down_"+str(sz_t))(x)
            shortCut = x
            x = gen_conv_gated_ds(output_dim=channelNum, kernel_size=3, strides=1, rate=1, padding='SAME', name="encode_conv_"+str(sz_t))(x)
            # apply residual
            x = keras.layers.add([shortCut, x])
            activations.append(x)
        
        # dilated conv
        # not done yet
        # x = dilate_block2(x, channelNum, 3, rate=1, name='re_en_dilated_1')
        x = gen_conv_gated_ds(output_dim=channelNum, kernel_size=3, strides=1, rate=1, padding='SAME', name="dilated_1")(x)
        x = gen_conv_gated_ds(output_dim=channelNum, kernel_size=3, strides=1, rate=2, padding='SAME', name="dilated_2")(x)
        x = gen_conv_gated_ds(output_dim=channelNum, kernel_size=3, strides=1, rate=4, padding='SAME', name="dilated_4")(x)
        x = gen_conv_gated_ds(output_dim=channelNum, kernel_size=3, strides=1, rate=8, padding='SAME', name="dilated_8")(x)
        x = gen_conv_gated_ds(output_dim=channelNum, kernel_size=3, strides=1, rate=16, padding='SAME', name="dilated_16")(x)

        # attention
        # not done yet
        mask_s = mask_input
        x, match, _ = apply_contextual_attention(x, mask_s, method=config.ATTENTION_TYPE, \
                                                           name='re_att_' + str(sz_t), dtype=dtype, conv_func=conv2)




                


