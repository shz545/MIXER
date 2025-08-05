import time
import numpy as np
import jax
import jax.numpy as jnp
import tensorflow as tf

results = []

# JAX GPU 檢查與矩陣乘法
jax_platforms = jax.devices()
jax_gpu_available = any(device.platform == 'gpu' for device in jax_platforms)

try:
    key = jax.random.PRNGKey(0)
    a = jax.random.normal(key, (1000, 1000))
    b = jax.random.normal(key, (1000, 1000))

    start_time = time.time()
    c = jnp.dot(a, b).block_until_ready()
    jax_time = time.time() - start_time

    results.append(f"✅ JAX matrix multiplication completed in {jax_time:.4f} seconds.")
    results.append(f"JAX GPU available: {'Yes' if jax_gpu_available else 'No'}")
except Exception as e:
    results.append(f"❌ JAX computation failed: {str(e)}")

# TensorFlow GPU 檢查與卷積運算
tf_gpu_available = tf.config.list_physical_devices('GPU')

try:
    input_data = tf.random.normal([1, 128, 128, 3])
    filters = tf.random.normal([5, 5, 3, 16])

    start_time = time.time()
    conv_result = tf.nn.conv2d(input_data, filters, strides=[1, 1, 1, 1], padding='SAME')
    tf_time = time.time() - start_time

    results.append(f"✅ TensorFlow convolution completed in {tf_time:.4f} seconds.")
    results.append(f"TensorFlow GPU available: {'Yes' if tf_gpu_available else 'No'}")
except Exception as e:
    results.append(f"❌ TensorFlow computation failed: {str(e)}")

# 印出結果
for line in results:
    print(line)