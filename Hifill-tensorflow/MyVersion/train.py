import yaml
from easydict import EasyDict as edict
import tensorflow as tf
from model import MyModel
from trainer import Trainer
import load_data

def load_yml(path):
    with open(path, 'r') as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
            print(config)
            return edict(config)
        except yaml.YAMLError as exc:
            print(exc)






if __name__ == "__main__":
    # load config.yml file
    config = load_yml('MyVersion/config1.yml')
    if config.GPU_ID != -1:
        gpu_ids = config.GPU_ID
    else:
        gpu_ids = [0]

    print('loading train&validation data...')

    '''    # load training data
    data_dir = config.TRAIN_LIST
    data_dir = './data/examples/train/'
    train_ds =  tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels=None,
        validation_split=0.2,
        subset="training",
        seed=123,
        shuffle=True,
        image_size=(512, 512),
        batch_size=config.BATCH_SIZE,
        color_mode="rgb"
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels=None,
        validation_split=0.2,
        subset="validation",
        seed=123,
        shuffle=True,
        image_size=(512, 512),
        batch_size=config.BATCH_SIZE,
        color_mode="rgb"
    )'''

    # def data path
    data_dir = './MyData/train/'
    # get dataset
    train_ds = load_data.create_dataset(data_dir, config.IMG_SHAPE, config.BATCH_SIZE)
    print('train_ds', train_ds.element_spec)

    model = MyModel("Mymodel", config)

    # def weight path
    dir_path = './model_weight'
    # trainer = Trainer(model, config, dir_path)
    trainer = Trainer(model, config)
    trainer.train(train_ds, config.CONTINUE_TRAIN, config.MAX_ITERS)