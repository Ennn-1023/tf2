import tensorflow as tf
from layers import Discriminator_block

# Our model class
class MyModel:
    def __init__(self, name):
        self.name = name
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

    def build_generator(self, img_input, mask_input):
        """
        Build the generator model
        """

        img_input = tf.keras.layers.Input(shape=(512, 512, 3))
        mask_input = tf.keras.layers.Input(shape=(512, 512, 1))

        # add masked image
        x = tf.concat([img_input, mask_input], axis=3) # (512, 512, 4)

        # sample output
        output = tf.keras.layers.Conv2D(3, (3, 3), padding='same')(x)
        return tf.keras.Model(inputs=[img_input, mask_input], outputs=output)
        


    def build_discriminator(self, training=True, nc=64):
        model = tf.keras.Sequential()
        model.add(Discriminator_block(nc=nc, training=training))
        return model
        
