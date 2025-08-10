import jax
import jax.numpy as jnp
import numpy as np
import jax.lax as lax
import optax
import time
from flax.training import train_state
from flax.training.train_state import TrainState
from flax.core import FrozenDict
from dataset import load_dataset
from utils import cutmix_data
from tqdm import tqdm  # 放在檔案最上方
from model import MlpMixer

import random

# 🔹 染色體結構
def sample_chromosome():
    return {
        "num_blocks": random.choice([4,6,8,10,12]),
        "patch_size": random.choice([4, 8]),
        "hidden_dim": random.choice([16, 32, 64, 128, 256]),
        "tokens_mlp_dim": random.choice([32, 64, 128, 256, 512]),
        "channels_mlp_dim": random.choice([32, 64, 128, 256, 512]),
        "dropout_rate": random.uniform(0.0, 0.5),
        "learning_rate": random.uniform(5e-4, 5e-2),
        "use_bn": random.choice([True])
    }

def mutate(chromosome):
    key = random.choice(list(chromosome.keys()))
    chromosome[key] = sample_chromosome()[key]
    return chromosome

def crossover(parent1, parent2):
    child = {}
    for key in parent1:
        child[key] = random.choice([parent1[key], parent2[key]])
    return child

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
        # Warmup phase
        def warmup_fn(_):
            return base_lr * step / warmup_steps

        def anneal_fn(_):
            # Remove warmup steps
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
    variables = model.init(rng, jnp.ones([1, 32, 32, 3]), train=True)
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
        schedule = schedule_fn  # 直接保留原始函數

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

# 🔹 單步訓練（加入 MixUp + BN 更新）
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

# 🔹 驗證步驟（BN 推論模式）
def eval_step(state, batch, smoothing=0.1):
    imgs, labels = batch
    variables = {'params': state.params, 'batch_stats': state.batch_stats}
    logits = state.apply_fn(variables, imgs, train=False, mutable=False)
    loss = cross_entropy_loss(logits, labels, smoothing)
    acc = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
    return {'loss': loss, 'accuracy': acc}

# 🔹 EarlyStopping 機制
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

# 🔹 資料集分割
def split_dataset(data, split_ratio=0.9):
    split_idx = int(len(data) * split_ratio)
    return data[:split_idx], data[split_idx:]

# 🔹 適合度函數（GGA）
def fitness(chromosome, rng):
    start_time = time.time()  # 開始計時
    tqdm.write("📊 開始評估")
    
    model = MlpMixer(
        num_classes=10,
        num_blocks=chromosome["num_blocks"],
        patch_size=chromosome["patch_size"],
        hidden_dim=chromosome["hidden_dim"],
        tokens_mlp_dim=chromosome["tokens_mlp_dim"],
        channels_mlp_dim=chromosome["channels_mlp_dim"],
        dropout_rate=chromosome["dropout_rate"],
        use_bn=chromosome["use_bn"]
    )

    batch_size = 128
    num_epochs = 12
    full_train_data = load_dataset(batch_size=batch_size, train=True)
    train_data, val_data = split_dataset(full_train_data, split_ratio=0.9)
    steps_per_epoch = len(train_data)
    warmup_steps = 2 * steps_per_epoch

    state, _ = create_train_state(
        rng, model,
        learning_rate=chromosome["learning_rate"],
        num_epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        warmup_steps=warmup_steps,
        weight_decay=5e-5
    )

    for epoch in range(num_epochs):
        for batch in tqdm(train_data, desc=f"Train Epoch {epoch+1}/{num_epochs}", leave=False):
            imgs, labels = batch
            mixed_imgs, y_a, y_b, lam = cutmix_data(np.array(imgs), np.array(labels))
            state, metrics = train_step(state, mixed_imgs, y_a, y_b, lam, labels)

    val_acc = 0
    for batch in tqdm(val_data, desc="Validating", leave=False):
        metrics = eval_step(state, batch)
        val_acc += float(metrics['accuracy'])

    end_time = time.time()  # 結束計時
    elapsed = end_time - start_time
    tqdm.write(f"⏱ 個體訓練時間：{elapsed:.2f} 秒")
    
    return val_acc / len(val_data), elapsed

# 🔹 GGA 主流程
def run_gga(pop_size=6, generations=5):
    rng = jax.random.PRNGKey(42)
    population = [sample_chromosome() for _ in range(pop_size)]

    tqdm.write("🔧 預熱 JAX 編譯器...")

    total_start_time = time.time()  # 整體開始計時

    for gen in range(generations):
        tqdm.write(f"\n📘 Generation {gen+1}\n")

        scores = []
        times = []
        for i, ind in enumerate(tqdm(population, desc=f"Evaluating Gen {gen+1}", leave=True)):
            tqdm.write(f"🧬 個體 {i+1} 染色體參數：{ind}")
            score, elapsed = fitness(ind, rng)
            scores.append(score)
            times.append(elapsed)
            tqdm.write(f"✅ 個體 {i+1} 驗證準確率: {score:.4f}")
            tqdm.write(f"⏱ 訓練時間: {elapsed:.2f} 秒")
            tqdm.write("--------------------------------------------------------------------------------------------------------------------------------------")

        sorted_pop = [x for _, x in sorted(zip(scores, population), key=lambda pair: pair[0], reverse=True)]
        parents = sorted_pop[:2]

        best_score = max(scores)
        avg_time = sum(times) / len(times)
        tqdm.write(f"🏅 Gen {gen+1} 最佳準確率: {best_score:.4f}")
        tqdm.write(f"🕒 Gen {gen+1} 平均訓練時間: {avg_time:.2f} 秒")

        new_population = []
        while len(new_population) < pop_size:
            child = crossover(parents[0], parents[1])
            child = mutate(child)
            new_population.append(child)
        population = new_population

    total_end_time = time.time()
    total_elapsed = total_end_time - total_start_time
    tqdm.write(f"\n⏳ 所有世代總訓練時間: {total_elapsed:.2f} 秒")

    best_chromosome = sorted_pop[0]
    tqdm.write("\n🏆 最佳染色體參數：")
    for k, v in best_chromosome.items():
        tqdm.write(f"{k}: {v}")
    return best_chromosome
