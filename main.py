# 模組載入
from train import train_with_config
import matplotlib.pyplot as plt
from model import MlpMixer
import os
import numpy as np
import argparse
import pickle

# 預設用 GPU
os.environ["JAX_PLATFORM_NAME"] = "gpu"

import jax
import jax.numpy as jnp

if any(device.platform == "gpu" for device in jax.devices()):
    print("✅ 使用 GPU 執行")
else:
    print("⚠️ 未偵測到 GPU，自動切換為 CPU 執行")
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    # 這時要重啟程式才會生效
    
def save_param_to_mem(param: np.ndarray, filename: str) -> None:
    """將參數存成 .mem 檔（Q8.8 格式）"""
    arr = np.array(param)
    arr_q88 = np.round(arr * 256).astype(np.int16).flatten()
    with open(filename, "w") as f:
        for val in arr_q88:
            f.write(f"{np.uint16(val):04X}\n")

def export_all_params_q88(params, folder="orig_kernel", prefix=""):
    os.makedirs(folder, exist_ok=True)
    for k, v in params.items():
        if isinstance(v, dict):
            export_all_params_q88(v, folder, prefix + k + "_")
        else:
            arr = np.array(v)
            fname = f"{folder}/{prefix}{k}.mem"
            save_param_to_mem(arr, fname)
            # 顯示參數資訊
            print(f"已儲存 {fname}，shape={arr.shape}, min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}")
            if arr.ndim == 2:
                print(f"  in_dim={arr.shape[0]}, out_dim={arr.shape[1]}")

# 類別名稱
dataset_name = "mnist"  # 直接訓練 mnist
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

def main():
    optimizer = "adamw"
    earlystop = "n"
    num_epochs = 30
    batch_size = 128

    default_config = {
        "num_blocks": 2,
        "patch_size": 4,
        "hidden_dim": 16,
        "tokens_mlp_dim": 32,
        "channels_mlp_dim": 32,
        "dropout_rate": 0.1,
        "learning_rate": 0.003,
        "use_bn": True
    }
    # 取得測試集 acc/loss 曲線
    test_accs, test_losses, model_params = train_with_config(
        default_config,
        num_epochs=num_epochs,
        batch_size=batch_size,
        earlystop=earlystop,
        dataset_name=dataset_name,
        optimizer=optimizer
    )
    # 儲存訓練後參數到 orig_kernel 資料夾
    export_all_params_q88(model_params, folder="orig_kernel")
    
    # 表格化顯示模型結構
    model = MlpMixer(
        num_blocks=default_config["num_blocks"],
        patch_size=default_config["patch_size"],
        hidden_dim=default_config["hidden_dim"],
        tokens_mlp_dim=default_config["tokens_mlp_dim"],
        channels_mlp_dim=default_config["channels_mlp_dim"],
        dropout_rate=default_config["dropout_rate"],   # <--- 加這行
        num_classes=len(classes)
    )
    dummy_input = jnp.ones((1, 32, 32, 1), dtype=jnp.float32)  # MNIST 單通道
    print(model.tabulate(
        jax.random.PRNGKey(0),
        dummy_input,
        False,  # 推論模式
        depth=6,
        console_kwargs={"width": 200}
    ))

if __name__ == "__main__":
    main()