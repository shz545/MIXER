import jax
import jax.numpy as jnp
import numpy as np
import jax.lax as lax
import optax
import time
import os
import pickle
import matplotlib.pyplot as plt
from flax.training import train_state
from flax.training.train_state import TrainState
from flax.core import FrozenDict
from dataset import load_dataset
from utils import cutmix_data,format_duration,plot_all_metrics
from tqdm import tqdm
from model import MlpMixer
import pickle  # 新增匯入 pickle 模組

# 🔹 擴充 TrainState 以支援 BatchNorm
class TrainState(train_state.TrainState):
    batch_stats: FrozenDict

# 🔹 Cross entropy loss（label smoothing）
def cross_entropy_loss(logits, labels, smoothing=0.1):
    num_classes = logits.shape[-1]
    one_hot = jax.nn.one_hot(labels, num_classes)
    soft_labels = one_hot * (1 - smoothing) + smoothing / num_classes
    return optax.softmax_cross_entropy(logits, soft_labels).mean()

def cosine_annealing_with_restarts_schedule(
    base_lr, warmup_steps, initial_period, period_mult=2, eta_min=0.0
):
    def schedule(step):
        def warmup_fn(_):
            return base_lr * step / warmup_steps

        def anneal_fn(_):
            s = step - warmup_steps
            period = initial_period
            accumulated = 0

            def cond_fn(vals):
                s, accumulated, period = vals
                return s >= accumulated + period

            def body_fn(vals):
                s, accumulated, period = vals
                return s, accumulated + period, period * period_mult

            s, accumulated, period = lax.while_loop(cond_fn, body_fn, (s, accumulated, period))
            t_cur = s - accumulated
            cosine = jnp.cos(jnp.pi * t_cur / period)
            return eta_min + 0.5 * (base_lr - eta_min) * (1 + cosine)

        return lax.cond(step < warmup_steps, warmup_fn, anneal_fn, operand=None)

    return schedule

def create_train_state(
    rng, model, learning_rate, num_epochs=None, steps_per_epoch=None,
    warmup_steps=0, weight_decay=5e-5, initial_period=500, period_mult=2,
    eta_min=0.0, optimizer="adamw", momentum=0.9
):
    variables = model.init(rng, jnp.ones([1, 32, 32, 1]), train=True)
    params = variables['params']
    batch_stats = variables.get('batch_stats', FrozenDict())

    if num_epochs and steps_per_epoch:
        warmup_steps = 2 * steps_per_epoch
        schedule_fn = cosine_annealing_with_restarts_schedule(
            base_lr=learning_rate,
            warmup_steps=warmup_steps,
            initial_period=initial_period,
            period_mult=period_mult,
            eta_min=eta_min
        )
        schedule = schedule_fn

        if optimizer.lower() == "adamw":
            tx = optax.adamw(schedule, weight_decay=weight_decay)
        elif optimizer.lower() == "sgd":
            tx = optax.chain(
                optax.trace(decay=momentum, nesterov=True),
                optax.scale_by_schedule(schedule),
                optax.scale(-1.0)
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")
    else:
        schedule = None
        if optimizer.lower() == "adamw":
            tx = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
        elif optimizer.lower() == "sgd":
            tx = optax.sgd(learning_rate=learning_rate, momentum=momentum, nesterov=True)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")

    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        batch_stats=batch_stats
    ), schedule

@jax.jit
def train_step(state, mixed_imgs, y_a, y_b, lam, labels, smoothing=0.1):
    def loss_fn(params):
        dropout_rng = jax.random.fold_in(jax.random.PRNGKey(0), state.step)
        variables = {'params': params, 'batch_stats': state.batch_stats}
        logits, new_model_state = state.apply_fn(
            variables,
            mixed_imgs,
            train=True,
            rngs={'dropout': dropout_rng},
            mutable=["batch_stats"]
        )
        loss_a = cross_entropy_loss(logits, y_a, smoothing)
        loss_b = cross_entropy_loss(logits, y_b, smoothing)
        loss = lam * loss_a + (1 - lam) * loss_b
        return loss, (logits, new_model_state)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits, new_model_state)), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=new_model_state["batch_stats"])
    acc = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
    return state, {'loss': loss, 'accuracy': acc}

def eval_step(state, batch, smoothing=0.1):
    imgs, labels = batch
    variables = {'params': state.params, 'batch_stats': state.batch_stats}
    logits = state.apply_fn(variables, imgs, train=False, mutable=False)
    loss = cross_entropy_loss(logits, labels, smoothing)
    acc = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
    return {'loss': loss, 'accuracy': acc}

class EarlyStopping:
    def __init__(self, patience=10, enabled=True):
        self.best_loss = float('inf')
        self.counter = 0
        self.patience = patience
        self.enabled = enabled

    def should_stop(self, val_loss):
        if not self.enabled:
            return False
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

def split_dataset(data, split_ratio=0.9):
    split_idx = int(len(data) * split_ratio)
    return data[:split_idx], data[split_idx:]

def train_with_config(config, num_epochs=10, batch_size=128, earlystop="n", dataset_name="mnist", optimizer="adamw"):
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

    full_train_data = load_dataset(batch_size=batch_size, train=True, dataset_name=dataset_name)
    train_data, val_data = split_dataset(full_train_data, split_ratio=0.9)
    test_data = load_dataset(batch_size=batch_size, train=False, dataset_name=dataset_name)
    
    tqdm.write(f"📦 訓練資料筆數: {len(train_data)}")
    steps_per_epoch = len(train_data) 
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = 5 * steps_per_epoch
    initial_period = 6 * steps_per_epoch
    
    schedule = cosine_annealing_with_restarts_schedule(
        base_lr=config["learning_rate"],
        warmup_steps=warmup_steps,
        initial_period=initial_period,
        period_mult=2,
        eta_min=0.00001
    )
    
    state, _ = create_train_state(
        rng,
        model,
        learning_rate=config["learning_rate"],
        num_epochs=None,
        steps_per_epoch=None,
        warmup_steps=warmup_steps,
        weight_decay=5e-5,
        initial_period=500,
        period_mult=2,
        eta_min=0.00001,
        optimizer=optimizer,
        momentum=0.9
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
            lr = float(schedule(current_step))
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

        if earlystop == "y":
            if early_stopping.should_stop(val_loss):
                print(f"⛔ Early stopping triggered at epoch {epoch+1}")
                break

    test_accs, test_losses = [], []
    test_loss, test_acc = 0, 0
    for batch in test_data:
        metrics = eval_step(state, batch)
        test_loss += float(metrics['loss'])
        test_acc += float(metrics['accuracy'])
    test_loss /= len(test_data)
    test_acc /= len(test_data)

    print(f"\n✅ Final Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")
    print(f"總耗時: {format_duration(time.time() - start_time)}")

    plot_all_metrics(train_accs, train_losses, val_accs, val_losses, lrs)
    
    # 儲存模型參數到 pkl
    with open("trained_model_params.pkl", "wb") as f:
        pickle.dump(state.params, f)
    print("✅ 已將模型參數存成 trained_model_params.pkl")
    
    return test_accs, test_losses, state.params