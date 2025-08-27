import numpy as np
import tensorflow as tf
from csd import graph_decompose

# hyper-param：延遲約束
dc = 2

# 1. 載入訓練好的 MLP-Mixer
model = tf.keras.models.load_model("mlp_mixer_model.h5")

# 2. 找到所有要做 DA 的層
layers_to_decompose = []
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.Dense) or "EinsumDense" in layer.__class__.__name__:
        W, b = layer.get_weights()
        din, dout = W.shape
        # 太小的層就略過
        if max(din, dout) < 16:  
            continue
        layers_to_decompose.append((layer.name, W, b))

# 3. 一層層跑 graph_decompose
for name, W, b in layers_to_decompose:
    print(f"Decomposing layer `{name}`, shape {W.shape} …")
    M1, M2 = graph_decompose(W, dc=dc)
    print(f"  → M1: {M1.shape}, M2: {M2.shape}")
    # 4. 存檔以便 HLS 或下一階段用
    np.save(f"{name}_M1.npy", M1)
    np.save(f"{name}_M2.npy", M2)

print("全部完成，M1/M2 已經存在工作目錄。")
