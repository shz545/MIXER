#python MIXER/main.py

# 模組載入
from train import run_gga, train_with_config

# 類別名稱
dataset_name = "cifar10"  # ✅ 可選 "cifar10" 或 "mnist"
if dataset_name == "cifar10":
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
elif dataset_name == "mnist":
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

def main():
    mode = "gga"  # ✅ 可選 "train" 或 "gga"
    trainornot = "y" # ✅ 可選 "y" 或 "n"
    optimizer = "adamw" # ✅ 可選 "adamw" 或 "sgd"
    earlystop = "n" # ✅ 可選 "y" 或 "n"
    num_epochs = 300
    pop_size = 10
    generations = 10
    batch_size = 128

    if mode == "train":
        default_config = {
            "num_blocks": 8,
            "patch_size": 4,
            "hidden_dim": 128,
            "tokens_mlp_dim": 64,
            "channels_mlp_dim": 512,
            "dropout_rate": 0.0,
            "learning_rate": 0.001,
            "use_bn": True
        }
        train_with_config(default_config, num_epochs=num_epochs, batch_size=batch_size, earlystop=earlystop, dataset_name=dataset_name, optimizer=optimizer)

    elif mode == "gga":
        best_config = run_gga(pop_size=pop_size, generations=generations, dataset_name=dataset_name, optimizer=optimizer) #pop_size 個體數(需>=2) , generations 世代數

        if trainornot == "y":
            print("\n🎯 使用最佳參數進行完整訓練")
            train_with_config(best_config, num_epochs=num_epochs, batch_size=batch_size, earlystop=earlystop, dataset_name=dataset_name, optimizer=optimizer)
        else:
            print("\n🎯 GGA結束 不進行完整訓練")

if __name__ == "__main__":
    main()