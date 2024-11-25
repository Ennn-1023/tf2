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
        self.Conv2D = keras.layers.Conv2D(self.output_dim, self.kernel_size, self.strides, dilation_rate=self.rate, 
                                 padding=self.padding, activation=self.activation, dtype=self.dtype,
                                 kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.05),
                                 bias_initializer=keras.initializers.Constant(0.0))
        self.Conv2D_DS = keras.layers.SeparableConv2D(filters=self.output_dim, kernel_size=3, strides=self.strides, 
                                          padding=self.padding, activation=self.activation, dtype = self.dtype,
                                          depthwise_initializer=keras.initializers.TruncatedNormal(stddev=0.05),
                                          pointwise_initializer=keras.initializers.TruncatedNormal(stddev=0.05),
                                          bias_initializer=keras.initializers.Constant(0.0))

    def call(self, inputs):
        # x1 = conv2d
        # x2 = conv2d_ds
        x1 = self.Conv2D(inputs)
        x1 = keras.layers.ELU()(x1)
        x2 = self.Conv2D_DS(inputs)
        x2 = keras.layers.Softmax()(x2)
        output = keras.layers.multiply([x1, x2])
        return output
class gen_deconv_gated_ds(keras.layers.Layer):
    def __init__(self, output_dim, kernel_size, strides=1, rate=1, padding='SAME'
                 , trainable=True, name='gated_deconv', dtype=tf.float32, **kwargs):
        super(gen_deconv_gated_ds, self).__init__(trainable, name, dtype, **kwargs)
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.strides = strides
        self.rate = rate
        self.padding = padding
        self.DecodeConv = gen_conv_gated_ds(output_dim=self.output_dim, kernel_size=self.kernel_size
                                            , strides=self.strides, rate=self.rate, padding=self.padding, name=self.name)
    def call(self, inputs):
        input_shape = inputs.get_shape().as_list()
        # resize input
        x = keras.layers.Resizing(height=input_shape[1]*2, width=input_shape[2]*2,interpolation='bilinear')(inputs)
        x = self.DecodeConv(x)
        return x

class dilate_block2(keras.layers.Layer): # not checked yet
    def __init__(self, input, output_dim, kernel_size, rate, name='dilate_block2', **kwargs):
        super(dilate_block2, self).__init__(name=name, **kwargs)
        self.input = input
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.rate = rate
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
    def __init__(self, dim, method, name='contextual_attention', dtype=tf.float32, **kwargs):
        super(contextual_attention_block, self).__init__(name=name, dtype=dtype, **kwargs)
        self.method = method
        self.ContextualAtt_layer = contextual_attention(method=self.method, name=self.name, rate=2, dtype=self.dtype)
        self.AttConv1 = gen_conv_gated_ds(output_dim=dim, kernel_size=3, strides=1, rate=1, padding='SAME', name="att_conv1")
        self.AttConv2 = gen_conv_gated_ds(output_dim=dim, kernel_size=3, strides=1, rate=1, padding='SAME', name="att_conv2")
    def call(self, inputs):
        x, mask_s = inputs
        x_hallu = x
        x, corres = self.ContextualAtt_layer([x, x, mask_s])
        x = self.AttConv1(x)
        x = keras.layers.Concatenate(axis=3)([x, x_hallu])
        x = self.AttConv2(x)
        
        return x, corres
     
class contextual_attention(keras.layers.Layer): # not checked yet
    def __init__(self, method='SOFT', kernel_size=3, rate=1, fuse_k=3, softmax_scale=10., fuse=True, name='contextual_attention', dtype=tf.float32, **kwargs):
        super(contextual_attention, self).__init__(name=name, dtype=dtype, **kwargs)
        self.method = method
        self.kernel_size = kernel_size
        self.rate = rate
        self.fuse_k = fuse_k
        self.softmax_scale = softmax_scale
        self.fuse = fuse
    def downsample(self, x, rate):
        shp = x.get_shape().as_list()
        print("shp:", shp)
        assert shp[1] % rate == 0 and shp[2] % rate == 0, 'height and width should be multiples of rate'
        shp[1], shp[2] = shp[1]//rate, shp[2]//rate # downsample height and width
        x = tf.image.extract_patches(images=x, sizes=[1, 1, 1, 1], strides=[1, rate, rate, 1], rates=[1, 1, 1, 1], padding='SAME')
        return keras.layers.Reshape(shp[1:])(x)
    
    def call(self, inputs):
        # input = [src, ref, mask]
        src = inputs[0]
        ref = inputs[1]
        mask = inputs[2]
        src_shape = src.get_shape().as_list()
        ref_shape = ref.get_shape().as_list()
        assert src_shape[1] == ref_shape[1] and \
            src_shape[3] == ref_shape[3], 'source and reference shape mismatch'
        batch_size = src.get_shape().as_list()[0]
        channelNum = src.get_shape().as_list()[3]
        rate = self.rate
        # raw features
        kernel = rate*2 -1
        # --- have to check this part ---
        # 要研究一下這裡的 dim_ordering 和變化

        raw_features = tf.image.extract_patches(images=ref, sizes=[1, kernel, kernel, 1], strides=[1, rate, rate, 1], rates=[1, 1, 1, 1], padding='SAME')
        print("raw_features after extract: ", raw_features.get_shape().as_list())
        raw_features = keras.layers.Reshape([-1, kernel, kernel, channelNum])(raw_features)
        raw_features = keras.layers.Permute([2, 3, 4, 1])(raw_features) # transpose to [batch, k, k, c, h*w]
        raw_features_lst = tf.split(raw_features, num_or_size_splits=batch_size, axis=0)

        # resize
        src = self.downsample(src, rate)
        ref = self.downsample(ref, rate)

        ss = src.get_shape().as_list()
        rs = ref.get_shape().as_list()
        # --- have to check this part ---
        
        features = tf.image.extract_patches(images=ref, sizes=[1, kernel, kernel, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='SAME')
        features = keras.layers.Reshape([-1, kernel, kernel, channelNum])(features)
        features = keras.layers.Permute([2, 3, 4, 1])(features) # transpose to [batch, k, k, c, h*w]
        features_lst = tf.split(features, num_or_size_splits=batch_size, axis=0)

        # mask
        mask = keras.layers.MaxPool2D(pool_size=(16,16), strides=(16,16), padding='SAME')(mask)
        mask = keras.layers.MaxPool2D(pool_size=(3,3), strides=(1,1), padding='SAME')(mask)
        mask = keras.layers.subtract([tf.constant(1.0, shape=mask.get_shape().as_list()), mask])
        print("mask shape in att64:", mask.get_shape().as_list())
        mask_lst = tf.split(mask, num_or_size_splits=batch_size, axis=0)

        y_lst, y_up_lst= [], []
        src_lst = tf.split(src, num_or_size_splits=batch_size, axis=0)
        fuse_weight = tf.reshape(tf.eye(self.fuse_k), [self.fuse_k, self.fuse_k, 1, 1])
        #fuse_weight = keras.layers.Reshape(target_shape=[self.fuse_k, self.fuse_k, 1, 1])(tf.eye(self.fuse_k)) # ?
        ref_feat = tf.divide(features, tf.maximum(tf.reduce_sum(tf.square(features), axis=[0, 1, 2]), 1e-8))
        src_lst = tf.split(src, num_or_size_splits=batch_size, axis=0)
        ref_feat_lst = tf.split(ref_feat, num_or_size_splits=batch_size, axis=0)
        y = []
        for src_i, ref_f in zip (src_lst, ref_feat_lst):
            y_i = tf.nn.conv2d(src_i, ref_f[0], strides=(1,1), padding='SAME')
            y.append(y_i)
        y = tf.concat(y, axis=0)

        if self.fuse:
            yi = keras.layers.Reshape([ss[1]*ss[2], rs[1]*rs[2], 1])(y)
            yi = tf.nn.conv2d(yi, fuse_weight, strides=(1,1), padding='SAME')
            yi = keras.layers.Reshape([ss[1], ss[2], rs[1], rs[2]])(yi)
            yi = keras.layers.Permute([2, 1, 4, 3])(yi)
            yi = keras.layers.Reshape([ss[1]*ss[2], rs[1]*rs[2], 1])(yi)
            yi = tf.nn.conv2d(yi, fuse_weight, strides=(1,1), padding='SAME')
            yi = keras.layers.Reshape([ss[2], ss[1], rs[2], rs[1]])(yi)
            yi = keras.layers.Permute([2, 1, 4, 3])(yi)
            y = yi
        y = keras.layers.Reshape([ss[1], ss[2], rs[1]*rs[2]])(y)

        if self.method == 'HARD':
            ym = tf.reduce_max(y, axis=-1, keepdims=True)
            y = y*mask
            coef = tf.cast(tf.greater_equal(y, tf.reduce_max(y, axis=-1, keepdims=True)), dtyp=self.dtype)
            y = tf.pow(coef * tf.divide(y, ym + 1e-04), 2)
        elif self.method == 'SOFT':
            y = keras.layers.Softmax(axis=-1)(y * mask * self.softmax_scale) * mask
        y = keras.layers.Reshape([ss[1], ss[2], rs[1]*rs[2]])(y)
        y_lst = tf.split(y, num_or_size_splits=batch_size, axis=0)
        y_up = []
        for y_i, raw_f in zip(y_lst, raw_features_lst):

            y_up.append(tf.nn.conv2d_transpose(y_i, raw_f[0], output_shape=[1]+src_shape[1:], strides=(1, rate, rate, 1), padding='SAME'))
        y_up = tf.concat(y_up, axis=0)
        out, correspondence = y_up, y
        out = keras.layers.Reshape([src_shape[1], src_shape[2], src_shape[3]])(out)
        return out, correspondence
        # for x, ref_feat, raw_ref_feat, mask in zip(src_lst, features_lst, raw_features_lst, mask_lst):
        #     ref_feat = ref_feat[0]
        #     ref_feat = tf.divide(ref_feat, tf.maximum(tf.reduce_sum(tf.square(ref_feat), axis=[0, 1, 2]), 1e-8))
        #     # y = keras.layers.Conv2D(ref_feat, kernel_size=3, strides=(1,1), padding='SAME')(x)
        #     y = tf.nn.conv2d(x, ref_feat, strides=(1,1), padding='SAME')
        #     if self.fuse:
        #         yi = keras.layers.Reshape([ss[1]*ss[2], rs[1]*rs[2], 1])(y)
        #         # yi = keras.layers.Conv2D(filters=fuse_weight, strides=(1,1), padding='SAME')(yi)
        #         yi = tf.nn.conv2d(yi, fuse_weight, strides=(1,1), padding='SAME')
        #         yi = keras.layers.Reshape([ss[1], ss[2], rs[1], rs[2]])(yi)
        #         yi = keras.layers.Permute([2, 1, 4, 3])(yi)
        #         yi = keras.layers.Reshape([ss[1]*ss[2], rs[1]*rs[2]], 1)(yi)
        #         # yi = keras.layers.Conv2D(filters=fuse_weight, strides=(1,1), padding='SAME')(yi)
        #         yi = tf.nn.conv2d(yi, fuse_weight, strides=(1,1), padding='SAME')
        #         yi = keras.layers.Reshape([ss[2], ss[1], rs[2], rs[1]])(yi)
        #         yi = keras.layers.Permute([2, 1, 4, 3])(yi)
        #         y = yi
        #     y = keras.layers.Reshape([ss[1], ss[2], rs[1]*rs[2]])(y)

        #     if self.method == 'HARD':
        #         ym = tf.reduce_max(y, axis=3, keepdims=True)
        #         y = y*mask
        #         coef = tf.cast(tf.greater_equal(y, tf.reduce_max(y, axis=3, keepdims=True)), dtyp=self.dtype)
        #         y = tf.pow(coef * tf.divide(y, ym + 1e-04), 2)
        #     elif self.method == 'SOFT':
        #         y = tf.nn.softmax(y * mask * self.softmax_scale, 3) * mask
        #     y.set_shape(1, src_shape[1], src_shape[2], ref_shape[1]*ref_shape[2])

        #     feats = raw_ref_feat[0]
        #     y_up =  tf.nn.conv2d_transpose(y, feats, output_shape=src_shape[1:], strides=(1, rate, rate, 1), padding='SAME')
        #     y_lst.append(y)
        #     y_up_lst.append(y_up)
        # out, correspondence = tf.concat(y_up_lst, axis=0), tf.concat(y_lst, axis=0)
        # out.set_shape([batch_size, src_shape[1], src_shape[2], src_shape[3]])
    
        # if dtype == tf.float32: skip this check
        # skip flow computation
        # not used in model
        # return out, correspondence

class Attention_layer(keras.layers.Layer):
    def __init__(self, dim, method, name='attention', dtype=tf.float32, **kwargs):
        super(Attention_layer, self).__init__(name=name, dtype=dtype, **kwargs)
        self.method = method
        self.AttDecodeConv1 = gen_conv_gated_ds(output_dim=dim, kernel_size=3, strides=1, rate=1, padding='SAME', name="att_decoder_conv_1")
        self.AttDecodeConv2 = gen_conv_gated_ds(output_dim=dim, kernel_size=3, strides=1, rate=1, padding='SAME', name="att_decoder_conv_2")
    def call(self, inputs):
        x_shape = inputs[0].get_shape().as_list()
        corres_shape = inputs[1].get_shape().as_list()
        rate = x_shape[1]// corres_shape[1]
        kernel = rate*2
        channelNum = x_shape[3]

        raw_feat = tf.image.extract_patches(inputs[0], sizes=[1, kernel, kernel, 1], 
                                            strides=[1, rate, rate, 1], rates=[1, 1, 1, 1], padding='SAME')
        raw_feat = keras.layers.Reshape([-1, kernel, kernel, channelNum])(raw_feat)
        raw_feat = keras.layers.Permute([2, 3, 4, 1])(raw_feat)
        raw_feat_lst = tf.split(raw_feat, num_or_size_splits=x_shape[0], axis=0) # split along batch axis

        y_score = []
        att_lst = tf.split(inputs[1], num_or_size_splits=x_shape[0], axis=0) # split along batch axis
        for feat, att in zip(raw_feat_lst, att_lst):
            y = tf.nn.conv2d_transpose(att, feat[0], [1] + x_shape[1:], strides=[1, rate, rate, 1], padding='SAME')
            y_score.append(y)
        out = tf.concat(y_score, axis=0)

        out = self.AttDecodeConv1(out)
        out = self.AttDecodeConv2(out)

        return out

def Build_Generator(input_shp=(512, 512), config=None, dtype=tf.float32):
    img_input = keras.layers.Input(shape=(input_shp[0], input_shp[1], 3), batch_size=config.BATCH_SIZE)
    mask_input = keras.layers.Input(shape=(input_shp[0], input_shp[1], 1), batch_size=config.BATCH_SIZE)
    xnow = keras.layers.concatenate([img_input, mask_input], axis=3) # (512, 512, 4)
    activations = [img_input]
    # encoder
    sz = input_shp[0]
    sz_t = input_shp[0]
    x = xnow
    channelNum = config.GEN_NC
    channelNum = max(4, channelNum // (sz_t // 512)) // 2
    while sz_t > config.BOTTLENECK_SIZE:
        channelNum *= 2
        sz_t //= 2
        # kkernal = 5 if sz_t == self.input_shape[0][0] else 3
        # 檢查一下這行原本有沒有用到
        x = gen_conv_gated_ds(output_dim=channelNum, kernel_size=3, strides=2, rate=1, padding='SAME', name="encoder_down_"+str(sz_t))(x)
        shortCut = x
        x = gen_conv_gated_ds(output_dim=channelNum, kernel_size=3, strides=1, rate=1, padding='SAME', name="encoder_conv_"+str(sz_t))(x)
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
    x, match = contextual_attention_block(x.get_shape().as_list()[-1], method=config.ATTENTION_TYPE, name='att_score' + str(sz_t), dtype=dtype)([x, mask_s])
    # decoder
    activations.pop(-1)
    while sz_t < sz//2:
        channelNum = channelNum//2
        sz_t *= 2
        x = gen_deconv_gated_ds(output_dim=channelNum, kernel_size=3, strides=1, rate=1, padding='SAME', name="decode_up_"+str(sz_t))(x)
        x = gen_conv_gated_ds(output_dim=channelNum, kernel_size=3, strides=1, rate=1, padding='SAME', name="decode_conv_"+str(sz_t))(x)
        x_att = activations.pop(-1)
        x_att = Attention_layer(x_att.get_shape().as_list()[-1], method=config.ATTENTION_TYPE, dtype=dtype, name='att_decode'+str(sz_t))([x_att, match])
        x = keras.layers.concatenate([x_att, x], axis=3)
    # decode to RGB 3 channels
    x = gen_deconv_gated_ds(output_dim=3, kernel_size=3, strides=1, rate=1, padding='SAME', name="decode_final")(x)
    x2 = tf.clip_by_value(x, -1.0, 1.0)
    return keras.models.Model(inputs=[img_input, mask_input], outputs=x2)