import tensorflow as tf
from tensorflow import keras
import time
# def trainer
class Trainer:
    def __init__(self, model, config = None):
        self.model = model # my model class
        self.config = config
        self.dis_optimizer = keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.9)
        self.gen_optimizer = keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.9)
        
    def discriminator_loss(self, real_output, fake_output):
        raise NotImplementedError
    def generator_loss(self, fake_output):
        raise NotImplementedError
    

    @tf.function
    def train_step(self, train_ds):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.model.generator(train_ds)
            real_output = self.model.discriminator(train_ds)
            fake_output = self.model.discriminator(generated_images)

            gen_loss = self.model.generator_loss(fake_output)
            disc_loss = self.model.discriminator_loss(real_output, fake_output)

    def train(dataset, epochs):
        for epoch in range(epochs):
            start = time.time()

            for image_batch in dataset:
                train_step(image_batch)

            # Produce images for the GIF as you go
            display.clear_output(wait=True)
            generate_and_save_images(generator,
                             epoch + 1,
                             seed)

            # Save the model every 15 epochs
            if (epoch + 1) % 15 == 0:
                checkpoint.save(file_prefix = checkpoint_prefix)

            print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
    

        
        
            