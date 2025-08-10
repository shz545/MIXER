#python MIXER/main.py

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import time
from tqdm import tqdm
# 模組載入
from model import MlpMixer
from dataset import load_dataset
from train import run_gga, create_train_state, train_step, eval_step, EarlyStopping, split_dataset
from utils import plot_all_metrics,cutmix_data

# CIFAR-10 類別名稱
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

def format_duration(seconds):
    mins, secs = divmod(int(seconds), 60)
    return f"{mins} 分 {secs} 秒"

def train_with_config(config, dataset_name="cifar10"):
    rng = jax.random.PRNGKey(0)
    start_time = time.time()

    model = MlpMixer(
        num_classes=10,
        num_blocks=config["num_blocks"],
        patch_size=config["patch_size"],
        hidden_dim=config["hidden_dim"],
        tokens_mlp_dim=config["tokens_mlp_dim"],
        channels_mlp_dim=config["channels_mlp_dim"],
        dropout_rate=config["dropout_rate"],
        use_bn=config["use_bn"]
    )

    batch_size = 128
    num_epochs = 250

    full_train_data = load_dataset(batch_size=batch_size, train=True, dataset_name=dataset_name)
    train_data, val_data = split_dataset(full_train_data, split_ratio=0.9)
    test_data = load_dataset(batch_size=batch_size, train=False, dataset_name=dataset_name)
    
    tqdm.write(f"📦 訓練資料筆數: {len(train_data)}")
    steps_per_epoch = len(train_data) #cifar10:351
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = 5 * steps_per_epoch
    initial_period = 6 * steps_per_epoch

    state, schedule = create_train_state(
        rng, model,
        learning_rate=config["learning_rate"],
        num_epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        warmup_steps=warmup_steps,
        weight_decay=5e-5
    )
    state, schedule = create_train_state(
    rng,
    model,
    learning_rate=config["learning_rate"],
    num_epochs=num_epochs,
    steps_per_epoch=steps_per_epoch,
    optimizer="adamw",  # "adamw" or "sgd"
    initial_period=initial_period,
    period_mult=2,
    eta_min=1e-5
    )

    train_accs, train_losses, val_accs, val_losses, lrs = [], [], [], [], []
    early_stopping = EarlyStopping(patience=5, enabled=False)

    print("🎬 訓練開始，計時開始")

    for epoch in range(num_epochs):
        epoch_train_loss, epoch_train_acc = 0, 0
        batch_count = 0
        epoch_start = time.time()

        for batch_idx, batch in enumerate(train_data):
            imgs, labels = batch
            mixed_imgs, y_a, y_b, lam = cutmix_data(np.array(imgs), np.array(labels))
            state, metrics = train_step(state, mixed_imgs, y_a, y_b, lam, labels)
            current_step = epoch * steps_per_epoch + batch_idx
            lr = float(schedule(current_step)) if schedule else config["learning_rate"]
            lrs.append(lr)

            epoch_train_loss += float(metrics['loss'])
            epoch_train_acc += float(metrics['accuracy'])
            batch_count += 1

            if epoch == 0 and batch_idx < 10:
                print(f"Step {current_step+1}, LR: {lr:.6f}")

        train_accs.append(epoch_train_acc / batch_count)
        train_losses.append(epoch_train_loss / batch_count)

        val_loss, val_acc = 0, 0
        for batch in val_data:
            metrics = eval_step(state, batch)
            val_loss += float(metrics['loss'])
            val_acc += float(metrics['accuracy'])
        val_loss /= len(val_data)
        val_acc /= len(val_data)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1} — Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accs[-1]:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, LR: {lr:.6f}")
        print(f"Epoch {epoch+1} 耗時: {format_duration(time.time() - epoch_start)}, Train_Acc多: {(train_accs[-1] - val_acc):.4f}")
        '''
        if early_stopping.should_stop(val_loss):
            print(f"⛔ Early stopping triggered at epoch {epoch+1}")
            break
        '''
    acc_list = [
    jnp.mean(jnp.argmax(model.apply({'params': state.params, 'batch_stats': state.batch_stats}, imgs, train=False), axis=-1) == labels)
    for imgs, labels in test_data
    ]
    test_acc = jnp.mean(jnp.array(acc_list))
    
    print(f"\n✅ Final Test Accuracy: {test_acc:.4f}")
    print(f"• 模型配置: \n"
          f"num_blocks: {config['num_blocks']} \n"
          f"patch_size: {config['patch_size']} \n"
          f"hidden_dim: {config['hidden_dim']} \n"
          f"tokens_mlp_dim: {config['tokens_mlp_dim']} \n"
          f"channels_mlp_dim: {config['channels_mlp_dim']} \n"
          f"dropout_rate: {config['dropout_rate']} \n"
          f"use_bn: {config['use_bn']}")
    print(f"總耗時: {format_duration(time.time() - start_time)}")

    plot_all_metrics(train_accs, train_losses, val_accs, val_losses, lrs)

def main():
    mode = "gga"  # ✅ 可選 "train" 或 "gga"

    if mode == "train":
        default_config = {
            "num_blocks": 8,
            "patch_size": 4,
            "hidden_dim": 256,
            "tokens_mlp_dim": 256,
            "channels_mlp_dim": 512,
            "dropout_rate": 0.014,
            "learning_rate": 0.02,
            "use_bn": True
        }
        train_with_config(default_config)

    elif mode == "gga":
        best_config = run_gga(pop_size=10, generations=10) #pop_size 個體數(需>=2) , generations 世代數
        print("\n🎯 使用最佳參數進行完整訓練")
        train_with_config(best_config)

if __name__ == "__main__":
    main()