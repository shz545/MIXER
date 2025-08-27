import numpy as np
import pickle
from csd import graph_decompose

# hyper-param：延遲約束
dc = 2

# 1. 載入訓練好的 MLP-Mixer（Flax 參數）
with open("mlp_mixer_params.pkl", "rb") as f:
    params = pickle.load(f)
# params 是 Flax 權重 dict

# 2. 找到所有要做 DA 的層（以 'Dense' 為例，可依你的 model 結構調整）
layers_to_decompose = []
for module_name in params:
    # 只處理有 kernel 權重的層（通常是 Dense）
    if "kernel" in params[module_name]:
        W = np.array(params[module_name]["kernel"])
        b = np.array(params[module_name].get("bias", np.zeros(W.shape[1])))
        din, dout = W.shape
        if max(din, dout) < 16:
            continue
        layers_to_decompose.append((module_name, W, b))

# 3. 一層層跑 graph_decompose
for name, W, b in layers_to_decompose:
    print(f"Decomposing layer `{name}`, shape {W.shape} …")
    M1, M2 = graph_decompose(W, dc=dc)
    print(f"  → M1: {M1.shape}, M2: {M2.shape}")
    # 4. 存檔以便 HLS 或下一階段用
    np.save(f"{name}_M1.npy", M1)
    np.save(f"{name}_M2.npy", M2)

print("全部完成，M1/M2 已經存在工作目錄。")