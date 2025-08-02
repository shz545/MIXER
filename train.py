import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from flax.core import FrozenDict
from dataset import load_dataset
from utils import mixup_data
from model import MlpMixer

# 🔹 擴充 TrainState 以支援 BatchNorm
class TrainState(train_state.TrainState):
    batch_stats: FrozenDict

# 🔹 Cross entropy loss（label smoothing）
def cross_entropy_loss(logits, labels, smoothing=0.1):
    num_classes = logits.shape[-1]
    one_hot = jax.nn.one_hot(labels, num_classes)
    soft_labels = one_hot * (1 - smoothing) + smoothing / num_classes
    return optax.softmax_cross_entropy(logits, soft_labels).mean()

# 🔹 建立訓練狀態（初始化 BN 狀態）
def create_train_state(rng, model, learning_rate, num_epochs=None, steps_per_epoch=None, warmup_steps=0, weight_decay=1e-2):
    print(f"🚀 Effective Warmup Steps: {warmup_steps}")
    variables = model.init(rng, jnp.ones([1, 32, 32, 3]), train=True)
    params = variables['params']
    batch_stats = variables.get('batch_stats', FrozenDict())

    if num_epochs and steps_per_epoch:
        total_steps = num_epochs * steps_per_epoch
        warmup_steps = min(warmup_steps, total_steps - 1)
        decay_steps = max(1, total_steps - warmup_steps)

        schedule = (optax.warmup_cosine_decay_schedule(
                        init_value=0.0,
                        peak_value=learning_rate,
                        warmup_steps=warmup_steps,
                        decay_steps=decay_steps,
                        end_value=0.0
                    ) if warmup_steps > 0 else
                    optax.cosine_decay_schedule(
                        init_value=learning_rate,
                        decay_steps=total_steps))

        tx = optax.adamw(schedule)
    else:
        schedule = None
        tx = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)

    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        batch_stats=batch_stats
    ), schedule

# 🔹 單步訓練（加入 MixUp + BN 更新）
@jax.jit
def train_step(state, batch, alpha=0.2, smoothing=0.1):
    imgs, labels = batch
    mixed_imgs, y_a, y_b, lam = mixup_data(imgs, labels, alpha)

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
        return lam * loss_a + (1 - lam) * loss_b, (logits, new_model_state)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits, new_model_state)), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=new_model_state["batch_stats"])
    acc = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
    return state, {'loss': loss, 'accuracy': acc}

# 🔹 驗證步驟（BN 推論模式）
def eval_step(state, batch, smoothing=0.1):
    imgs, labels = batch
    variables = {'params': state.params, 'batch_stats': state.batch_stats}
    logits = state.apply_fn(variables, imgs, train=False, mutable=False)
    loss = cross_entropy_loss(logits, labels, smoothing)
    acc = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
    return {'loss': loss, 'accuracy': acc}

# 🔹 EarlyStopping 機制
class EarlyStopping :
    def __init__(self, patience=5, enabled=True):
        self.best_loss = float('inf')
        self.counter = 0
        self.patience = patience
        self.enabled = enabled

    def should_stop(self, val_loss):
        if not self.enabled:
            return False  # 強制不啟用 EarlyStopping
        
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

# 🔹 資料集分割
def split_dataset(data, split_ratio=0.9):
    split_idx = int(len(data) * split_ratio)
    return data[:split_idx], data[split_idx:]