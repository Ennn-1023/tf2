import tensorflow as tf
from model import MyModel


import matplotlib.pyplot as plt


def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

model = MyModel("Mymodel", config)
dir_path = './model_weight'
model.discriminator.load_weights(dir_path + '/discriminator')
model.generator.load_weights(dir_path + '/generator')

