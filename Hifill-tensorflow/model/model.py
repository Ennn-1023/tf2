

import tensorflow as tf
from layers import Discriminator_block, gen_conv, gen_deconv, gen_deconv_gated, gen_conv_gated, gen_conv_gated_ds, \
    gen_deconv_gated_ds, gen_conv_gated_slice, gen_deconv_gated_slice


def get_conv_op(conv_type):
    #gen_conv = ops.GenConvGated()
    #print(conv_type, 'ds')
    if conv_type == 'none':
        conv = gen_conv
        deconv = gen_deconv
    elif conv_type == 'regular':
        conv = gen_conv_gated
        deconv = gen_deconv_gated
    elif conv_type == 'ds':
        conv = gen_conv_gated_ds
        deconv = gen_deconv_gated_ds
    elif conv_type == 'slice':
        conv = gen_conv_gated_slice
        deconv = gen_deconv_gated_slice
    else:
        raise('wrong conv type ' + conv_type)
    return conv, deconv

# Our model class
class MyModel:
    def __init__(self, name):
        self.name = name
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

    def build_generator(self, img_input, mask_input, config=None):
        """
        Build the generator model
        """

        img_input = tf.keras.layers.Input(shape=(512, 512, 3))
        mask_input = tf.keras.layers.Input(shape=(512, 512, 1))

        # add masked image
        x = tf.concat([img_input, mask_input], axis=3) # (512, 512, 4)
        # conv and deconv for stage-2
        conv2, deconv2 = get_conv_op(config.REFINE_CONV_TYPE)

        # sample output
        output = tf.keras.layers.Conv2D(3, (3, 3), padding='same')(x)
        return tf.keras.Model(inputs=[img_input, mask_input], outputs=output)
        


    def build_discriminator(self, training=True, nc=64):
        model = tf.keras.Sequential()
        model.add(Discriminator_block(nc=nc, training=training))
        return model
        
    

    def build_static_graph(self, real, config, mask=None, name='val'):
        raise NotImplementedError

    def build_inference_graph(self, real, mask, config=None, reuse=False, is_training=False, dtype=tf.float32):
        raise NotImplementedError