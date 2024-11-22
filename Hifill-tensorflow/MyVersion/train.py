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
    config = load_yml('MyVersion/config.yml')

    print('loading train&validation data...')

    # get dataset
    train_ds = load_data.create_dataset(config.TRAIN_PATH, config.IMG_SHAPE, config.BATCH_SIZE)
    print('train_ds', train_ds.element_spec)

    model = MyModel("Mymodel", config)
    # model.generator.build(input_shape=(config.BATCH_SIZE, 512, 512, 3))
    # model.generator.summary()
    # model.discriminator.summary()
    # def weight path
    dir_path = './model_weight'
    log_path = './train_log'
    # trainer = Trainer(model, config, dir_path)
    trainer = Trainer(model, config)
    trainer.train(train_ds, epochs=config.MAX_ITERS, dir_path=dir_path, log_path=log_path,continue_training=config.CONTINUE_TRAIN)
    # batch size 調整
    # 同一地點的圖片 training
    # 用相同做test
    # 印出 discrimnator accuracy