import os
import tensorflow as tf

def load_image(image_path, image_size):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.resize(image, image_size)
    return image


def load_data(original_dir, mask_dir, fixed_dir, image_size):
    original_images = []
    masks = []
    fixed_images = []

    # Get the filenames of the images
    # ensure the order of (original, mask, fixed) is the same
    filenames = sorted(os.listdir(fixed_dir))

    for filename in filenames:
        orig_path = os.path.join(original_dir, filename)
        mask_path = os.path.join(mask_dir, filename)
        fixed_path = os.path.join(fixed_dir, filename)

        if os.path.exists(orig_path) and os.path.exists(mask_path):
            original_images.append(load_image(orig_path, image_size))
            masks.append(load_image(mask_path, image_size))
            fixed_images.append(load_image(fixed_path, image_size))

    return tf.data.Dataset.from_tensor_slices((tf.stack(original_images), tf.stack(masks), tf.stack(fixed_images)))

def preprocess_data(data):
    """
    預處理函數，將資料集中每個元素的遮罩圖像轉換為二值圖像。
    
    參數:
        data: dict, 包含原始圖像、遮罩圖像和修復後圖像的字典。
    
    返回:
        dict, 預處理過的字典，遮罩圖像已經被轉換為二值圖像。
    """
    data['original_images'] = data['original_images'] / 255.0
    data['masks'] = convert_mask(data['masks'])  # Apply convert_mask to the masks
    data['fixed_images'] = data['fixed_images'] / 255.0
    return data

def create_dataset(original_dir, mask_dir, fixed_dir, image_size, batch_size):
    dataset = load_data(original_dir, mask_dir, fixed_dir, image_size)
    dataset = dataset.map(preprocess_data)
    dataset = dataset.shuffle(1000).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
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

    # 由於輸入圖像是二值圖像（白色和黑色），可以簡化為直接比較是否為白色
    # 假設像素值已被正規化在 [0, 1]，即白色為 1，黑色為 0
    gray_mask = gray_mask / 255.0
    binary_mask = tf.where(gray_mask > 0.5, 1.0, 0.0)

    return binary_mask

