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

def preprocess_data(original, mask, fixed):
    original = original / 255.0
    mask = mask / 255.0
    fixed = fixed / 255.0
    return {'original_images': original, 'masks': mask, 'fixed_images': fixed}

def create_dataset(original_dir, mask_dir, fixed_dir, image_size, batch_size):
    dataset = load_data(original_dir, mask_dir, fixed_dir, image_size)
    # dataset = dataset.map(preprocess_data)
    dataset = dataset.shuffle(buffer_size=1000).batch(batch_size)
    return dataset