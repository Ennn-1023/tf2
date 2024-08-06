import yaml
from easydict import EasyDict as edict
import tensorflow as tf
from ..model.model import MyModel
from ..trainer.trainer import Trainer

def load_yml(path):
    with open(path, 'r') as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
            print(config)
            return edict(config)
        except yaml.YAMLError as exc:
            print(exc)


def convert_mask(mask):
    """
    將 RGB 的 mask 轉換為單通道的二值 mask，白色部分為 1，黑色部分為 0。

    參數:
        mask: Tensor, 大小為 (512, 512, 3) 的 RGB mask 圖像。

    返回:
        Tensor, 大小為 (512, 512, 1) 的二值 mask。
    """
    # 計算灰度值，這是由於 RGB 到灰度的轉換通常按這樣的權重：0.299 R + 0.587 G + 0.114 B
    gray_mask = tf.reduce_mean(mask, axis=-1, keepdims=True)

    # 由於輸入圖像是二值圖像（白色和黑色），可以簡化為直接比較是否為白色
    # 假設像素值已被正規化在 [0, 1]，即白色為 1，黑色為 0
    gray_mask = gray_mask / 255.0
    binary_mask = tf.where(gray_mask > 0.5, 1.0, 0.0)

    return binary_mask



if __name__ == "__main__":
    # load config.yml file
    config = load_yml('config1.yml')
    if config.GPU_ID != -1:
        gpu_ids = config.GPU_ID
    else:
        gpu_ids = [0]

    print('loading train&validation data...')

    # load training data
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
    )


    model = MyModel("Mymodel")

    # def weight path
    dir_path = './model_weight'
    trainer = Trainer(model, config, dir_path)

    trainer.train(train_ds, config.CONTINUE_TRAIN, config.MAX_ITERS )