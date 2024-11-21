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
        self.channelNum = input.get_shape()[3]
    def call(self, input):
        channelNum = self.channelNum
        x = gen_conv_gated_ds(output_dim=channelNum, kernel_size=3, strides=1, rate=1, padding='SAME', name="dilated_1")(input)
        x = gen_conv_gated_ds(output_dim=channelNum, kernel_size=3, strides=1, rate=2, padding='SAME', name="dilated_2")(x)
        x = gen_conv_gated_ds(output_dim=channelNum, kernel_size=3, strides=1, rate=4, padding='SAME', name="dilated_4")(x)
        x = gen_conv_gated_ds(output_dim=channelNum, kernel_size=3, strides=1, rate=8, padding='SAME', name="dilated_8")(x)
        x = gen_conv_gated_ds(output_dim=channelNum, kernel_size=3, strides=1, rate=16, padding='SAME', name="dilated_16")(x)
        return x

class contextual_attention_block(keras.layers.Layer): # not checked yet
    def __init__(self, method, name='contextual_attention', dtype=tf.float32, conv_func=None, **kwargs):
        super(contextual_attention, self).__init__(name=name, dtype=dtype, **kwargs)
        self.method = method
        self.name = name
        self.dtype = dtype
        self.conv_func = conv_func
        self.size = input.get_shape().as_list()[1]
        self.channelNum = input.get_shape().as_list()[3]
    def call(self, input):
        size = self.size
        channelNum = self.channelNum
        x, mask_s = input
        x_hallu = x
        x, corres = contextual_attention([x, x, mask_s], method=self.method, name=self.name, dtype=self.dtype)
        x = gen_conv_gated_ds(output_dim=channelNum, kernel_size=3, strides=1, rate=1, padding='SAME', name="att_conv1")(x)
        x = keras.layers.Concatenate(axis=3)([x, x_hallu])
        x = gen_conv_gated_ds(output_dim=channelNum, kernel_size=3, strides=1, rate=1, padding='SAME', name="att_conv2")(x)
        
        return x, corres
     
class contextual_attention(keras.layers.Layer): # not checked yet
    def __init__(self, src_shape, ref_shape, method='SOFT', kernel_size=3, rate=1, fuse_k=3, softmax_scale=10., fuse=True, name='contextual_attention', dtype=tf.float32, **kwargs):
        super(contextual_attention, self).__init__(name=name, dtype=dtype, **kwargs)
        self.src_shape = src_shape
        self.ref_shape = ref_shape
        self.method = method
        self.name = name
        self.dtype = dtype
        self.kernel_size = kernel_size
        self.rate = rate
        self.fuse_k = fuse_k
        self.softmax_scale = softmax_scale
        self.fuse = fuse
        self.channelNum = input.get_shape().as_list()[3]
        assert self.src.get_shape().as_list()[0] == self.ref.get_shape().as_list()[0] and \
            self.src.get_shape().as_list()[3] == self.ref.get_shape().as_list()[3], 'source and reference shape mismatch'
        self.batch_size = self.src.get_shape().as_list()[0]
    def downsample(self, x, rate):
        shp = x.get_shape().as_list()
        assert shp[1] % rate == 0 and shp[2] % rate == 0, 'height and width should be multiples of rate'
        shp[1], shp[2] = shp[1]//rate, shp[2]//rate # downsample height and width
        x = tf.image.extract_patches(images=x, sizes=[1, rate, rate, 1], strides=[1, rate, rate, 1], rates=[1, 1, 1, 1], padding='SAME')
        return keras.layers.Reshape(shp[1:])(x)
    
    def call(self, input):
        # input = [src, ref, mask]
        src = input[0]
        ref = input[1]
        mask = input[2]
        src_shape = self.src_shape
        ref_shape = self.ref_shape
        batch_size = self.batch_size
        channelNum = self.channelNum
        rate = self.rate
        # raw features
        kernel = rate*2 -1
        # --- have to check this part ---
        # 要研究一下這裡的 dim_ordering 和變化
        raw_features = tf.image.extract_patches(images=ref, sizes=[1, kernel, kernel, 1], strides=[1, 1, 1, 1], rates=[1, rate, rate, 1], padding='SAME')
        raw_features = keras.layers.Reshape([batch_size, -1, kernel, kernel, channelNum])(raw_features)
        raw_features = keras.layers.Permute([2, 3, 4, 1])(raw_features) # transpose to [batch, k, k, c, h*w]
        raw_features_lst = tf.split(raw_features, num_or_size_splits=batch_size, axis=0)

        # resize
        src = self.downsample(src, rate)
        ref = self.downsample(ref, rate)

        src_shape = src.get_shape().as_list()
        ref_shape = ref.get_shape().as_list()
        ss = src_shape # downsampled shape
        rs = ref_shape # downsampled shape
        # --- have to check this part ---
        
        features = tf.image.extract_patches(images=ref, sizes=[1, kernel, kernel, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='SAME')
        features = keras.layers.Reshape([-1, kernel, kernel, channelNum])(features)
        features = keras.layers.Permute([2, 3, 4, 1])(features) # transpose to [batch, k, k, c, h*w]
        features_lst = tf.split(features, num_or_size_splits=batch_size, axis=0)

        # mask
        mask = keras.layers.MaxPool2D(pool_size=(16,16), strides=(16,16), padding='SAME')(mask)
        mask = keras.layers.MaxPool2D(pool_size=(3,3), strides=(1,1), padding='SAME')(mask)
        mask = keras.layers.subtract([tf.constant(1, shape=mask.get_shape().as_list()), mask])
        mask_lst = tf.split(mask, num_or_size_splits=batch_size, axis=0)

        y_lst, y_up_lst= [], []
        src_lst = tf.split(src, num_or_size_splits=batch_size, axis=0)
        fuse_weight = keras.layers.Reshape(target_shape=[self.fuse_k, self.fuse_k, 1, 1])(tf.eye(self.fuse_k)) # ?
        for x, ref_feat, raw_ref_feat, mask in zip(src_lst, features_lst, raw_features_lst, mask_lst):
            ref_feat = ref_feat[0]
            ref_feat = tf.divide(ref_feat, tf.maximum(tf.reduce_sum(tf.square(ref_feat), axis=[0, 1, 2]), 1e-8))
            y = keras.layers.Conv2D(ref_feat, kernel_size=3, strides=(1,1), padding='SAME')(x)
            if self.fuse:
                yi = keras.layers.Reshape([ss[1]*ss[2], rs[1]*rs[2], 1])(y)
                yi = keras.layers.Conv2D(filters=fuse_weight, strides=(1,1), padding='SAME')(yi)
                yi = keras.layers.Reshape([ss[1], ss[2], rs[1], rs[2]])(yi)
                yi = keras.layers.Permute([2, 1, 4, 3])(yi)
                yi = keras.layers.Reshape([ss[1]*ss[2], rs[1]*rs[2]], 1)(yi)
                yi = keras.layers.Conv2D(filters=fuse_weight, strides=(1,1), padding='SAME')(yi)
                yi = keras.layers.Reshape([ss[2], ss[1], rs[2], rs[1]])(yi)
                yi = keras.layers.Permute([2, 1, 4, 3])(yi)
                y = yi
            y = keras.layers.Reshape([ss[1], ss[2], rs[1]*rs[2]])(y)

            if self.method == 'HARD':
                ym = tf.reduce_max(y, axis=3, keepdims=True)
                y = y*mask
                coef = tf.cast(tf.greater_equal(y, tf.reduce_max(y, axis=3, keepdims=True)), dtyp=self.dtype)
                y = tf.pow(coef * tf.divide(y, ym + 1e-04), 2)
            elif self.method == 'SOFT':
                y = tf.nn.softmax(y * mask * self.softmax_scale, 3) * mask
            y.set_shape(1, src_shape[1], src_shape[2], ref_shape[1]*ref_shape[2])

            if self.dtype == tf.float32:
                offset = tf.argmax(y, axis=3, output_type=tf.float32)
                offsets.append(offset)
            feats = raw_ref_feat[0]
            # y_up = tf.nn.conv2d_transpose
            y_up =  tf.nn.conv2d_transpose(y, feats, output_shape=src_shape[1:], strides=(1, rate, rate, 1), padding='SAME')
            y_lst.append(y)
            y_up_lst.append(y_up)
        out, correspondence = tf.concat(y_up_lst, axis=0), tf.concat(y_lst, axis=0)
        out.set_shape([batch_size, src_shape[1], src_shape[2], src_shape[3]])
    
        # if dtype == tf.float32: skip this check
        # skip flow computation
        # not used in model
        return out, correspondence





# define the generator model
class generator(keras.models.Model):
    def __init__(self, input_shape=[512, 512], config=None):
        super().__init__()
        self.input_shape = [input_shape.copy().append(3), input_shape.copy().append(1)]
        self.config = config

    def call(self, input, training=True):
        img_input = input[0]
        mask_input = input[1]
        xnow = keras.layers.concatenate([img_input, mask_input], axis=3) # (512, 512, 4)
        activations = [img_input]
        # encoder
        sz = self.input_shape[0][0]
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
        x, match = contextual_attention_block(x, mask_s, method=self.config.ATTENTION_TYPE,
                                                 name='re_att_' + str(sz_t), dtype=self.dtype)

        # decoder
        activations.pop(-1)
        while sz_t < sz//2:
            channelNum = channelNum//2
            sz_t *= 2
            x = gen_deconv_gated_ds(output_dim=channelNum, kernel_size=3, strides=2, rate=1, padding='SAME', name="decode_up_"+str(sz_t))(x)
            shortCut = x
            x = gen_conv_gated_ds(output_dim=channelNum, kernel_size=3, strides=1, rate=1, padding='SAME', name="decode_conv_"+str(sz_t))(x)
            x = keras.layers.add([shortCut, x])
            # x_att = apply_attention(x, activations.pop(-1), method=self.config.ATTENTION_TYPE, name='att_' + str(sz_t), dtype=self.dtype)


        


                


