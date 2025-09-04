import tensorflow as tf
import jax.numpy as jnp

def preprocess(image, label):
    image = tf.image.resize_with_crop_or_pad(image, 40, 40)
    image = tf.image.random_crop(image, [32, 32, 3])
    image = tf.image.random_flip_left_right(image)
    image = tf.cast(image, tf.float32) / 255.0
    return image, tf.squeeze(label)

def preprocess_mnist(image, label):
    image = tf.image.resize(image, [32, 32])
    # 不要轉成 RGB，直接保留灰階
    image = tf.cast(image, tf.float32) / 255.0
    return image, tf.squeeze(label)

def load_dataset(batch_size=16, train=True, dataset_name="cifar10"):
    if dataset_name == "cifar10":
        (x, y), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        data = (x, y) if train else (x_test, y_test)
        ds = tf.data.Dataset.from_tensor_slices(data)
        ds = ds.map(preprocess).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    elif dataset_name == "mnist":
        (x, y), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x = x[..., None]  # MNIST 是灰階，要加 channel 維度
        x_test = x_test[..., None]
        data = (x, y) if train else (x_test, y_test)
        ds = tf.data.Dataset.from_tensor_slices(data)
        ds = ds.map(preprocess_mnist).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return [(jnp.array(imgs), jnp.array(labels)) for imgs, labels in ds]