import tensorflow as tf
import jax.numpy as jnp

def preprocess(image, label):
    image = tf.expand_dims(image, axis=-1)              # 灰階軸補上
    image = tf.image.grayscale_to_rgb(image)            # 灰階轉 RGB
    image = tf.image.resize(image, [32, 32])            # 如果你模型是吃 32x32
    image = tf.cast(image, tf.float32) / 255.0
    return image, tf.squeeze(label)

def load_dataset(batch_size=16, train=True):
    (x, y), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    data = (x, y) if train else (x_test, y_test)
    ds = tf.data.Dataset.from_tensor_slices(data)
    ds = ds.map(preprocess).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return [(jnp.array(imgs), jnp.array(labels)) for imgs, labels in ds]
