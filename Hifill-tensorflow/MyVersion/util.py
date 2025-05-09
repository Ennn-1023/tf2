import cv2
import tensorflow as tf
from model import MyModel
import load_data
import argparse
from tqdm import tqdm
from easydict import EasyDict as edict
import yaml
import os

import matplotlib.pyplot as plt

def load_yml(path):
    with open(path, 'r') as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
            print(config)
            return edict(config)
        except yaml.YAMLError as exc:
            print(exc)
            exit(1)

def load_test_data(fName):
    fixed = tf.image.decode_png(tf.io.read_file('./MyData/test/origin/'+fName))
    fixed = tf.image.resize(fixed, [512, 512])
    mask = tf.image.decode_png(tf.io.read_file('./MyData/test/mask/'+fName))
    mask = tf.image.resize(mask, [512, 512])

    fixed = fixed / 127.5 - 1.0
    mask = load_data.convert_mask(mask)
    fixed = tf.expand_dims(fixed, 0)
    mask = tf.expand_dims(mask, 0)
    return [fixed, mask]

def generate_and_save_images(model, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)
    for i in range(predictions.shape[0]):
        plt.subplot(1, 2, 1)
        plt.imshow(test_input[0][i] * 0.5 + 0.5)
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(predictions[i] * 0.5 + 0.5)
        plt.axis('off')
        plt.show()

class InferenceModel:
    def __init__(self, model_name='InferenceModel', config_path=None, weights_path=None, batch_size=1):
        assert config_path is not None, 'config_path should not be None'
        assert weights_path is not None, 'weights_path should not be None'
        self.config = load_yml(config_path)
        self.config.BATCH_SIZE = batch_size
        self.model = MyModel(model_name, self.config)
        self._load_model(weights_path)

    def _load_model(self, weights_path):
        self.model.generator.load_weights(weights_path)
    
    def inference_loop(self, input_ds, output_dir):
        with tqdm(total=len(input_ds), desc="inference", unit="batch") as pbar:
            for image_batch in input_ds:
                # generate images
                generated_images = self.model.generator([image_batch['fixed_images'], image_batch['masks']], training=False)
                # save images

                for i in range(generated_images.shape[0]):
                    img = generated_images[i].numpy()
                    img = (img + 1) * 127.5
                    img = img.astype('uint8')
                    cv2.imwrite(os.path.join(output_dir, image_batch['image_name'][i]), img)
                pbar.update(1)
        
        print("Inference completed. Generated images saved to:", output_dir)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='MyVersion/config1.yml', help='Path to the config file')
    parser.add_argument('-w', '--weight', type=str, default='model_weight/generator', help='Path to the weights file')
    parser.add_argument('-i', '--input', type=str, help='Path to the input dir')
    parser.add_argument('-o', '--output', type=str, help='Path to the output dir')
    parser.add_argument('-b', '--batch_size', type=int, default=1, help='Batch size for inference')
    opt, _ = parser.parse_known_args()
    
    model = InferenceModel(config_path=opt.config, weights_path=opt.weight, batch_size=opt.batch_size)
    # load dataset
    input_ds = load_data.create_dataset(opt.input, (512, 512), 1, key='validation', from_csv=True, train=False)
    # create output dir if not exist
    if not os.path.exists(opt.output):
        os.makedirs(opt.output)
    # inference loop
    model.inference_loop(input_ds, opt.output)
    
    


    # config = load_yml('MyVersion/config1.yml')
    # config.BATCH_SIZE = 1
    # model = MyModel("Mymodel", config)
    # dir_path = './model_weight/generator'
    # model.generator.load_weights(dir_path)
    # # load fixed image and mask
    # test_input = load_test_data('image.png')
    # generate_and_save_images(model.generator, test_input)
