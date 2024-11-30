

import tensorflow as tf
from .layers import Discriminator_block
from .generator import Build_Generator

# Our model class
class MyModel:
    def __init__(self, name, config=None):
        self.name = name
        self.generator = Build_Generator(config.IMG_SHAPE, config)
        self.discriminator = self.build_discriminator()

    def build_discriminator(self, training=True, nc=64):
        model = tf.keras.Sequential()
        model.add(Discriminator_block('discriminator_block',nc=nc, training=training))
        return model
        
