import tensorflow as tf
import jax.numpy as jnp

MNIST_MEAN = [0.1307]
MNIST_STD = [0.3081]
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.247, 0.243, 0.261]

def preprocess(image, label, train=True, dataset='mnist'):
    if dataset == 'mnist':
        image = tf.expand_dims(image, -1) if image.shape.rank == 2 else image  # ➕ channel 維度
        image = tf.image.grayscale_to_rgb(image)  # ➡️ 轉成 3 通道
        image = tf.image.resize(image, [32, 32])  # ⬆️ 升成與 CIFAR 相同大小

        if train:
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, 0.2)
            image = tf.image.random_contrast(image, 0.8, 1.2)

        image = tf.cast(image, tf.float32) / 255.0
        image = (image - MNIST_MEAN[0]) / MNIST_STD[0]
    else:
        if train:
            image = tf.image.resize_with_crop_or_pad(image, 40, 40)
            image = tf.image.random_crop(image, [32, 32, 3])
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, 0.2)
            image = tf.image.random_contrast(image, 0.8, 1.2)
            image = tf.image.random_saturation(image, 0.8, 1.2)
        else:
            image = tf.image.resize(image, [32, 32])
        image = tf.cast(image, tf.float32) / 255.0
        image = (image - CIFAR10_MEAN) / CIFAR10_STD

    return image, tf.squeeze(label)

def load_dataset(batch_size=64, train=True, dataset='mnist'):
    if dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = x_train[..., None]
        x_test = x_test[..., None]
        data = (x_train, y_train) if train else (x_test, y_test)
    else:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        data = (x_train, y_train) if train else (x_test, y_test)

    ds = tf.data.Dataset.from_tensor_slices(data)
    ds = ds.map(lambda img, lbl: preprocess(img, lbl, train, dataset), num_parallel_calls=tf.data.AUTOTUNE)
    if train:
        ds = ds.shuffle(1000)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return [(jnp.array(imgs), jnp.array(labels)) for imgs, labels in ds]