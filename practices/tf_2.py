import tensorflow as tf
from tensorflow import keras
import numpy as np

import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape)

# normalize
x_train, x_test = x_train/255.0, x_test/255.0

# model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

# print(model.summary())

# functional API
inputs = keras.Input(shape=(28,28))
flatten = keras.layers.Flatten()
dense1 = keras.layers.Dense(units=128, activation='relu')
dense2 = keras.layers.Dense(units=10)
dense2_2 = keras.layers.Dense(units=1)

layer1_output = flatten(inputs)
layer2_output = dense1(layer1_output)
outputs = dense2(layer2_output)
outputs2 = dense2_2(layer2_output)

model = keras.Model(inputs=inputs, outputs=outputs, name='functional_moodel')
print(model.summary())