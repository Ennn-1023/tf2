import tensorflow as tf


# Our model class
class MyModel:
    def __init__(self, name):
        self.name = name

    def build_generator(self, img_input, mask_input):
        """
        Build the generator model
        
        """

        img_input = tf.keras.layers.Input(shape=(512, 512, 3))
        mask_input = tf.keras.layers.Input(shape=(512, 512, 1))

        # add masked image
        x = tf.concat([img_input, mask_input], axis=3) # (512, 512, 4)

        # sample output
        output = tf.keras.layers.Conv2D(3, (3, 3), padding='same')(input)
        return tf.keras.Model(inputs=input, outputs=output)
        


    def build_discriminator(self, x, reuse=False, training=True, nc=64):
        raise NotImplementedError

    def build_graph_with_losses(self, real, config, training=True, summary=False, reuse=False):
        '''
        Build the model
        '''

    def build_static_graph(self, real, config, mask=None, name='val'):
        raise NotImplementedError

    def build_inference_graph(self, real, mask, config=None, reuse=False, is_training=False, dtype=tf.float32):
        raise NotImplementedError