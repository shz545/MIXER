import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import time

from model import MlpMixer
from dataset import load_dataset
from train import create_train_state, train_step, eval_step, EarlyStopping, split_dataset
from eval import evaluate
from utils import plot_all_metrics

classes = [str(i) for i in range(10)]  # MNIST 是 0~9 的數字

def format_duration(seconds):
    mins, secs = divmod(int(seconds), 60)
    return f"{mins} 分 {secs} 秒"


def main():
    rng = jax.random.PRNGKey(0)
    start_time = time.time()

    model = MlpMixer(
        num_classes=10,
        num_blocks=6,
        patch_size=4,
        hidden_dim=64,  
        tokens_mlp_dim=128,
        channels_mlp_dim=256,
        dropout_rate=0.3
    )
    learning_rate = 0.003
    batch_size = 64
    num_epochs = 100

    full_train_data = load_dataset(batch_size=batch_size, train=True)
    train_data, val_data = split_dataset(full_train_data, split_ratio=0.9)
    test_data = load_dataset(batch_size=batch_size, train=False)

    steps_per_epoch = len(train_data)
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = min(int(0.03 * total_steps), total_steps - 1)
    weight_decay = 1e-4

    state, schedule = create_train_state(
        rng, model,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay
    )

    train_accs, train_losses, val_accs, val_losses, lrs = [], [], [], [], []
    early_stopping = EarlyStopping(patience=5)

    print("訓練開始，計時開始")

    for epoch in range(num_epochs):
        epoch_train_loss, epoch_train_acc = 0, 0
        batch_count = 0
        epoch_start = time.time()

        for batch_idx, batch in enumerate(train_data):
            state, metrics = train_step(state, batch, weight_decay=weight_decay)
            current_step = epoch * steps_per_epoch + batch_idx
            lr = float(schedule(current_step)) if schedule else 0.001
            lrs.append(lr)

            epoch_train_loss += float(metrics['loss'])
            epoch_train_acc += float(metrics['accuracy'])
            batch_count += 1

            if epoch == 0 and batch_idx < 10:
                print(f"Step {current_step}, LR: {lr:.6f}")

        train_accs.append(epoch_train_acc / batch_count)
        train_losses.append(epoch_train_loss / batch_count)

        # ➕ 驗證集評估
        val_loss, val_acc = 0, 0
        for batch in val_data:
            metrics = eval_step(state.params, batch, state.apply_fn, state.batch_stats)
            val_loss += float(metrics['loss'])
            val_acc += float(metrics['accuracy'])
        val_loss /= len(val_data)
        val_acc /= len(val_data)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1} — Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accs[-1]:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, LR: {lr:.6f}")
        print(f"Epoch {epoch+1} 耗時: {format_duration(time.time() - epoch_start)}, Acc差: {(train_accs[-1] - val_acc):.4f}")

        #if early_stopping.should_stop(val_loss):
        #    print(f"⛔ Early stopping triggered at epoch {epoch+1}")
        #    break

    test_acc = evaluate(model, state.params, state.batch_stats, test_data)
    print(f"\n✅ Final Test Accuracy: {test_acc:.4f}")
    print(f"總耗時: {format_duration(time.time() - start_time)}")

    plot_all_metrics(train_accs, train_losses, val_accs, val_losses, lrs)

if __name__ == "__main__":
    main()