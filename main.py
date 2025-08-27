#python MIXER/main.py
      
# 模組載入
import jax                                                        # JAX 核心庫，用來控制隨機性和加速數值運算
import jax.numpy as jnp
from train import run_gga, train_with_config
from model import MlpMixer
import os
import numpy as np
import flax
import pickle
# 類別名稱
dataset_name = "mnist"  # ✅ 可選 "cifar10" 或 "mnist"
if dataset_name == "cifar10":
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
elif dataset_name == "mnist":
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

def save_param_to_mem(param, filename):
    arr = np.array(param)
    arr_q88 = np.round(arr * 256).astype(np.int16).flatten()
    with open(filename, "w") as f:
        for val in arr_q88:
            f.write(f"{np.uint16(val):04X}\n")

def export_all_params_q88(params, folder="kernel", prefix=""):
    os.makedirs(folder, exist_ok=True)
    for k, v in params.items():
        if isinstance(v, dict):
            export_all_params_q88(v, folder, prefix + k + "_")
        else:
            fname = f"{folder}/{prefix}{k}.mem"
            save_param_to_mem(v, fname)
            print(f"已儲存 {fname}，shape={np.array(v).shape}")
            
def main():
    mode = "train"  # ✅ 可選 "train" 或 "gga"
    trainornot = "y" # ✅ 可選 "y" 或 "n"
    optimizer = "adamw" # ✅ 可選 "adamw" 或 "sgd"
    earlystop = "n" # ✅ 可選 "y" 或 "n"
    num_epochs = 50
    pop_size = 10
    generations = 10
    batch_size = 128

    if mode == "train":
        default_config = {
            "num_blocks": 3,
            "patch_size": 4,
            "hidden_dim": 128,
            "tokens_mlp_dim": 64,
            "channels_mlp_dim": 512,
            "dropout_rate": 0.1,
            "learning_rate": 0.001,
            "use_bn": False
        }
        
        train_with_config(default_config, num_epochs=num_epochs, batch_size=batch_size, earlystop=earlystop, dataset_name=dataset_name, optimizer=optimizer)
        
        model = MlpMixer(
            num_classes=10,
            num_blocks=default_config["num_blocks"],
            patch_size=default_config["patch_size"],
            hidden_dim=default_config["hidden_dim"],
            tokens_mlp_dim=default_config["tokens_mlp_dim"],
            channels_mlp_dim=default_config["channels_mlp_dim"],
            dropout_rate=default_config["dropout_rate"],
            use_bn=default_config["use_bn"]
        )
        dummy_input = jnp.ones((1, 32, 32, 3), dtype=jnp.float32)
        variables = model.init(jax.random.PRNGKey(0), dummy_input, False)
        params = variables["params"]
        export_all_params_q88(params)
        # 儲存 Flax 參數
        with open("mlp_mixer_params.pkl", "wb") as f:
            pickle.dump(params, f)
        print("已儲存 Flax 訓練參數到 mlp_mixer_params.pkl")
        print(model.tabulate(
            jax.random.PRNGKey(0), 
            dummy_input, 
            False,
            depth=6,
            console_kwargs={"width": 200}
            ))

    elif mode == "gga":
        best_config = run_gga(pop_size=pop_size, generations=generations, dataset_name=dataset_name, optimizer=optimizer) #pop_size 個體數(需>=2) , generations 世代數

        if trainornot == "y":
            print("\n🎯 使用最佳參數進行完整訓練")
            train_with_config(best_config, num_epochs=num_epochs, batch_size=batch_size, earlystop=earlystop, dataset_name=dataset_name, optimizer=optimizer)
        else:
            print("\n🎯 GGA結束 不進行完整訓練")

if __name__ == "__main__":
    main()