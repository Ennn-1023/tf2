import os
import tensorflow as tf
import cv2
import pandas as pd
from tqdm import tqdm

import cv2
import tensorflow as tf
import os
import matplotlib.pyplot as plt


def create_dataset(dir_path, image_size, batch_size, key='train', from_csv=False, train=True):
    if from_csv:
        dataset = load_data_from_csv(dir_path, image_size[0:2], batch_size, key=key, train=train)
    else:
        dataset = load_data(dir_path, image_size[0:2], batch_size)
    return dataset

def preprocess_train_data(data, inpainted=False):
    """
    預處理函數，將資料集中每個元素的遮罩圖像轉換為二值圖像。
    
    參數:
        data: dict, 包含原始圖像、遮罩圖像和修復後圖像的字典。
    
    返回:
        dict, 預處理過的字典，遮罩圖像已經被轉換為二值圖像。
    """
    data['original_images'] = tf.io.read_file(data['original_images'])
    data['original_images'] = tf.image.decode_jpeg(data['original_images'], channels=3)
    data['original_images'] = tf.image.resize(data['original_images'], (512, 512))
    data['original_images'] = data['original_images'] / 127.5 - 1.0

    data['masks'] = tf.io.read_file(data['masks'])
    data['masks'] = tf.image.decode_jpeg(data['masks'], channels=1)
    data['masks'] = tf.image.resize(data['masks'], (512, 512))
    data['masks'] = convert_mask(data['masks'])  # Apply convert_mask to the masks
    
    if inpainted:
        data['fixed_images'] = tf.io.read_file(data['fixed_images'])
        data['fixed_images'] = tf.image.decode_jpeg(data['fixed_images'], channels=3)
        data['fixed_images'] = tf.image.resize(data['fixed_images'], (512, 512))
        data['fixed_images'] = data['fixed_images'] / 127.5 - 1.0
    else:
        inverted_mask = 1.0 - data['masks']  # 把 1->0, 0->1，讓 1 表示保留
        data['fixed_images'] = data['original_images'] * inverted_mask  # broadcasting 自動處理 channel

    
    return data

def preprocess_infer_data(data):
    data['masks'] = tf.io.read_file(data['masks'])
    data['masks'] = tf.image.decode_jpeg(data['masks'], channels=1)
    data['masks'] = tf.image.resize(data['masks'], (512, 512))
    data['masks'] = convert_mask(data['masks'])  # Apply convert_mask to the masks

    data['fixed_images'] = tf.io.read_file(data['fixed_images'])
    data['fixed_images'] = tf.image.decode_jpeg(data['fixed_images'], channels=3)
    data['fixed_images'] = tf.image.resize(data['fixed_images'], (512, 512))
    data['fixed_images'] = data['fixed_images'] / 127.5 - 1.0

    data['image_name'] = data['image_name']
    return data

def load_data_from_csv(csv_path, image_size = (512, 512), batch_size = 4, key='train', train=True):
    """
    從 CSV 文件中加載圖像數據集，並將其轉換為 TensorFlow 數據集格式。
    
    參數:
        csv_path: str, CSV 文件的路徑。
        image_size: tuple, 圖像大小，默認為 (512, 512)。
        batch_size: int, 批次大小，默認為 4。
    
    返回:
        dataset: tf.data.Dataset, TensorFlow 數據集對象。
    """
    # 讀取 CSV 文件
    df = pd.read_csv(csv_path)
    assert key in ('train', 'validation', 'all'), "key must be 'train', 'validation' or 'all'"
    if key != 'all':
        df = df[df['partition'] == key]

    image_paths = df['image_path'].values
    mask_paths = df['mask_path'].values
    fixed_paths = df['fixed_path'].values
    data_root = os.path.dirname(csv_path)
    if train:
        # 創建 TensorFlow 數據集
        dataset = tf.data.Dataset.from_tensor_slices(([os.path.join(data_root, l) for l in image_paths], 
                                                    [os.path.join(data_root, l) for l in mask_paths],
                                                    [os.path.join(data_root, l) for l in fixed_paths]))
        dataset = dataset.map(lambda orig, mask, fixed: {'original_images': orig, 'masks': mask, 'fixed_images': fixed})
        dataset = dataset.map(preprocess_train_data)
        dataset = dataset.prefetch(buffer_size = tf.data.experimental.AUTOTUNE)
        dataset = dataset.shuffle(buffer_size = 3000).batch(batch_size, drop_remainder = True)
    else: # inference mode
        dataset = tf.data.Dataset.from_tensor_slices(([os.path.join(data_root, l) for l in mask_paths],
                                                    [os.path.join(data_root, l) for l in fixed_paths],
                                                    [name for name in image_paths]))
        dataset = dataset.map(lambda mask, fixed, name: {'masks': mask, 'fixed_images': fixed, 'image_name': name})
        dataset = dataset.map(preprocess_infer_data)
        dataset = dataset.batch(batch_size, drop_remainder = True)


    return dataset

def load_data(image_path, image_size = (512, 512), batch_size = 4):
    origin_path = os.path.join(image_path, 'origin')
    mask_path = os.path.join(image_path, 'mask')
    fixed_path = os.path.join(image_path, 'fixed')

    origin = tf.keras.preprocessing.image_dataset_from_directory(
        origin_path,
        labels = None,
        label_mode = None,
        image_size = image_size,
        batch_size = None,
        shuffle = False,
    )
    mask = tf.keras.preprocessing.image_dataset_from_directory(
        mask_path,
        labels = None,
        label_mode = None,
        image_size = image_size,
        batch_size = None,
        shuffle = False,
    )
    fixed = tf.keras.preprocessing.image_dataset_from_directory(
        fixed_path,
        labels = None,
        label_mode = None,
        image_size = image_size,
        batch_size = None,
        shuffle = False,
    )
    dataset = tf.data.Dataset.zip((origin, mask, fixed))
    dataset = dataset.map(lambda orig, mask, fixed: {'original_images': orig, 'masks': mask, 'fixed_images': fixed})
    dataset = dataset.map(preprocess_train_data)
    dataset = dataset.prefetch(buffer_size = tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size = 3000).batch(batch_size, drop_remainder = True)

    return dataset


# def preprocess_data(data):
#     """
#     預處理函數，將資料集中每個元素的遮罩圖像轉換為二值圖像。
    
#     參數:
#         data: dict, 包含原始圖像、遮罩圖像和修復後圖像的字典。
    
#     返回:
#         dict, 預處理過的字典，遮罩圖像已經被轉換為二值圖像。
#     """
#     data['original_images'] = data['original_images'] / 127.5 - 1.0
#     data['masks'] = convert_mask(data['masks'])  # Apply convert_mask to the masks
#     data['fixed_images'] = data['fixed_images'] / 127.5 - 1.0
#     return data

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

    # 假設像素值已被正規化在 [0, 1]，即白色為 1，黑色為 0
    binary_mask = tf.where(gray_mask > 0.5, 1.0, 0.0)

    return binary_mask
