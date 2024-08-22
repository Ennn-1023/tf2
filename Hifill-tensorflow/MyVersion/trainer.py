import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import time


def gan_wgan_loss(pos, neg, name='gan_loss'):
    d_loss = tf.reduce_mean(neg) - tf.reduce_mean(pos)
    g_loss = -tf.reduce_mean(neg)
    return g_loss, d_loss

def random_interpolates(pos, neg):
    epsilon = tf.compat.v1.random_uniform(shape=[pos.get_shape().as_list()[0], 1, 1, 1],
            minval=0.,maxval=1., dtype = tf.float32)
    X_hat = pos + epsilon * (neg - pos)
    return X_hat

def gradients_penalty(interpolates_global, dout_global, mask):
    grad_D_X_hat = tf.gradients(dout_global, [interpolates_global])[0]
    red_idx = np.arange(1, len(interpolates_global.get_shape().as_list())).tolist()
    # slopes = tf.sqrt(tf.reduce_sum(tf.square(grad_D_X_hat), reduction_indices=red_idx)) # for older version
    slopes = tf.sqrt(tf.reduce_sum(tf.square(grad_D_X_hat), axis=red_idx))
    gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
    return gradient_penalty

# def trainer
class Trainer:
    def __init__(self, model, config = None):
        self.model = model # my model class
        self.config = config
        self.dis_optimizer = keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.9)
        self.gen_optimizer = keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.9)
        

    def compute_losses(self, gen_output, dis_output, interps, D_interps, dataset, masks):
        losses = {}
        coarse_alpha = self.config.COARSE_ALPHA
        preprocessed_images = dataset['images']
        real = dataset['original_images']
        losses['l1_loss'] = coarse_alpha*tf.reduce_mean(tf.abs(real - preprocessed_images)*masks)
        losses['l1_loss'] = tf.reduce_mean(tf.abs(real - gen_output)*masks)

        losses['ae_loss'] = coarse_alpha * tf.reduce_mean(tf.abs(real - preprocessed_images) * (1.-masks))
        losses['ae_loss'] += tf.reduce_mean(tf.abs(real - gen_output)* (1.-masks) )
        losses['ae_loss'] /= tf.reduce_mean(1.-masks)
         # gan loss
        dis_output
        D_real, D_fake = tf.split(dis_output, 2)
        g_loss, d_loss = gan_wgan_loss(D_real, D_fake, name='gan_loss')
        losses['g_loss'] = g_loss
        losses['d_loss'] = d_loss

        # apply gp
        gp_loss = gradients_penalty(interps, D_interps, mask=masks)
        losses['gp_loss'] = self.config.WGAN_GP_LAMBDA * gp_loss
        losses['d_loss'] = losses['d_loss'] + losses['gp_loss']
        losses['g_loss'] = self.config.GAN_LOSS_ALPHA * losses['g_loss']
        losses['g_loss'] += self.config.L1_LOSS_ALPHA * losses['l1_loss']
        losses['g_loss'] += self.config.AE_LOSS_ALPHA * losses['ae_loss']
        return losses
    

    @tf.function
    def train_step(self, train_ds):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
            
            generated_images = self.model.generator([train_ds['fixed_images'], train_ds['masks']], training=True)
            
            x = train_ds['original_images']*(1.-train_ds['masks'])
            fake_patched = generated_images * train_ds['masks'] + x * (1.-train_ds['masks']) # ?
            real_fake = tf.concat([train_ds['original_images'], fake_patched], axis=0)
            if self.config.GAN_WITH_MASK:
                real_fake = tf.concat([real_fake, tf.tile(train_ds['masks'], [self.config.BATCH_SIZE*2, 1, 1, 1])], axis=3)
            real_fake = self.model.discriminator([train_ds['original_images'], generated_images], training=True)
            interps = random_interpolates(train_ds['original_images'], fake_patched)
            D_interps = self.discriminator(interps, reuse=True, nc=self.config.DIS_NC)
            
            # compute losses
            losses = self.compute_losses(generated_images, real_fake, interps, D_interps, train_ds)
            
            grad_gen = gen_tape.gradient(losses['g_loss'], self.model.generator.trainable_variables)
            grad_dis = dis_tape.gradient(losses['d_loss'], self.model.discriminator.trainable_variables)
            self.gen_optimizer.apply_gradients(zip(grad_gen, self.model.generator.trainable_variables))
            self.dis_optimizer.apply_gradients(zip(grad_dis, self.model.discriminator.trainable_variables))

    def save(self, dir_path):
        self.model.generator.save_weights(dir_path + '/generator')
        self.model.discriminator.save_weights(dir_path + '/discriminator')


    def train(self, train_ds, dir_path, epochs, continue_training = False):
        '''
        Train the model
        params:
        dataset: training dataset
        dir_path: directory to load/save the model
        epochs: number of epochs to train
        continue_training: whether to continue training from a previous checkpoint
        '''

        # load the model if continue_training is True
        if continue_training:
            self.model.generator.load_weights(dir_path + '/generator')
            self.model.discriminator.load_weights(dir_path + '/discriminator')

        # train the model
        for epoch in range(epochs):
            start = time.time()

            for image_batch in train_ds:
                self.train_step(image_batch)

            # Save the model every 15 epochs
            if (epoch + 1) % 15 == 0:
                self.save()

            print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
            

        
        
            