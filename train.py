import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

# 計算 cross entropy loss（用 softmax + one-hot 編碼）
def cross_entropy_loss(logits, labels, smoothing=0.1):
    num_classes = logits.shape[-1]
    one_hot = jax.nn.one_hot(labels, num_classes)
    soft_labels = one_hot * (1 - smoothing) + smoothing / num_classes
    return optax.softmax_cross_entropy(logits, soft_labels).mean()


# 建立訓練狀態（包含 optimizer 與 LR schedule）
def create_train_state(rng, model, learning_rate, num_epochs=None, steps_per_epoch=None, warmup_steps=0, weight_decay=1e-2):
    # 初始化模型參數
    params = model.init(rng, jnp.ones([1, 32, 32, 3]), train=True)  # 假設輸入大小為 CIFAR-10 格式

    schedule = None
    if num_epochs is not None and steps_per_epoch is not None:
        total_steps = num_epochs * steps_per_epoch
        warmup_steps = min(warmup_steps, total_steps - 1)
        decay_steps = max(1, total_steps - warmup_steps)

        # 使用 warmup + cosine decay 作為學習率排程
        if warmup_steps > 0:
            schedule = optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=learning_rate,
                warmup_steps=warmup_steps,
                decay_steps=decay_steps,
                end_value=0.0
            )
        else:
            schedule = optax.cosine_decay_schedule(
                init_value=learning_rate,
                decay_steps=total_steps
            )
        tx = optax.adamw(schedule)

        # 預覽前幾步的 learning rate（debug 用）
        print("📊 Learning rate schedule preview:")
        for i in range(0, min(20, total_steps), max(1, total_steps // 20)):
            print(f"Step {i}: lr = {float(schedule(i)):.6f}")
    else:
        # 無排程的 fallback（固定 learning rate）
        tx = optax.adamw(learning_rate=schedule, weight_decay=weight_decay)  # ✅ 動態使用傳入的 weight_decay

    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )
    return state, schedule  # ⬅️ 回傳兩個物件：訓練狀態 & 學習率函數

@jax.jit
def train_step(state, batch):
    imgs, labels = batch

    def loss_fn(params):
        dropout_rng = jax.random.fold_in(jax.random.PRNGKey(0), state.step)
        logits = state.apply_fn(params, imgs, train=True, rngs={'dropout': dropout_rng})
        loss = cross_entropy_loss(logits, labels)
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
    return state, {'loss': loss, 'accuracy': accuracy}