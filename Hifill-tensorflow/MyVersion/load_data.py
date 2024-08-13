import os
import tensorflow as tf
import cv2
from tqdm import tqdm

import cv2
import tensorflow as tf
import os

def load_image(image_path, image_size):
    print (image_path)
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3, dtype=tf.float32)
    image = tf.image.resize(image, image_size[0:2])
    return image

def parse_image_paths(orig_path, mask_path, fixed_path, image_size):
    original_image = load_image(orig_path, image_size)
    mask_image = load_image(mask_path, image_size)
    fixed_image = load_image(fixed_path, image_size)
    return original_image, mask_image, fixed_image

def split(original, mask, fixed):
    return {'original_images': original, 'masks': mask, 'fixed_images': fixed}

def load_data(original_dir, mask_dir, fixed_dir, image_size, batch_size):
    filenames = sorted(os.listdir(fixed_dir))
    
    orig_paths = [os.path.join(original_dir, fname) for fname in filenames]
    mask_paths = [os.path.join(mask_dir, fname) for fname in filenames]
    fixed_paths = [os.path.join(fixed_dir, fname) for fname in filenames]

    # Create a Dataset from the image paths
    dataset = tf.data.Dataset.from_tensor_slices((orig_paths, mask_paths, fixed_paths))

    # Map the parse function to load and preprocess the images
    dataset = dataset.map(lambda orig_path, mask_path, fixed_path: parse_image_paths(orig_path, mask_path, fixed_path, image_size))
    dataset = dataset.map(split)

    # Shuffle, batch, and prefetch the dataset
    dataset = dataset.shuffle(buffer_size=len(filenames))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset

def preprocess_data(data):
    """
    預處理函數，將資料集中每個元素的遮罩圖像轉換為二值圖像。
    
    參數:
        data: dict, 包含原始圖像、遮罩圖像和修復後圖像的字典。
    
    返回:
        dict, 預處理過的字典，遮罩圖像已經被轉換為二值圖像。
    """
    data['original_images'] = data['original_images'] / 127.5 - 1.0
    data['masks'] = convert_mask(data['masks'])  # Apply convert_mask to the masks
    data['fixed_images'] = data['fixed_images'] / 127.5 - 1.0
    return data

def create_dataset(original_dir, mask_dir, fixed_dir, image_size, batch_size):
    dataset = load_data(original_dir, mask_dir, fixed_dir, image_size, batch_size)
    dataset = dataset.map(preprocess_data)
    return dataset

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
