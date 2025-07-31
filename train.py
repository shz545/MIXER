import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

# 📘 Cross-entropy loss function
def cross_entropy_loss(logits, labels):
    one_hot = jax.nn.one_hot(labels, num_classes=logits.shape[-1])
    return optax.softmax_cross_entropy(logits, one_hot).mean()

# 🎯 Accuracy metric
def compute_accuracy(logits, labels):
    predictions = jnp.argmax(logits, axis=-1)
    return jnp.mean(predictions == labels)

# 📈 Learning rate schedule
def create_learning_rate_schedule(warmup_steps, total_steps, base_lr):
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=base_lr,
        warmup_steps=warmup_steps,
        decay_steps=total_steps - warmup_steps,
        end_value=1e-5,
    )

# 🏗️ Initialize TrainState with proper parameter creation
def create_train_state(rng, model, learning_rate, warmup_steps, total_steps):
    dummy_input = jnp.ones([1, 32, 32, 3])
    
    # 初始化變數集合（可能包含params / batch_stats 等）
    variables = model.init(rng, dummy_input)

    # ✨ 如果需要動態建立參數（懶初始化），可執行一次 forward
    model.apply(variables, dummy_input)  # 若你沒用 BatchNorm / Dropout，這行就夠

    params = variables["params"]

    lr_schedule = create_learning_rate_schedule(warmup_steps, total_steps, learning_rate)
    tx = optax.adamw(learning_rate=lr_schedule)

    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )

# 🔁 Training step with JIT and gradient updates
@jax.jit
def train_step(state, batch):
    imgs, labels = batch

    def loss_fn(params):
        logits = state.apply_fn({'params': params}, imgs)
        loss = cross_entropy_loss(logits, labels)
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    acc = compute_accuracy(logits, labels)
    
    return state, {
        'loss': loss,
        'accuracy': acc,
    }