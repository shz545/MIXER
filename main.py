import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import time
from model import MlpMixer
from dataset import load_dataset
from train import create_train_state, train_step
from eval import evaluate
from utils import plot_metrics,plot_metric

classes = [str(i) for i in range(10)]  # MNIST 是 0~9 的數字

#def preprocess(image):
    #return jnp.array(image, dtype=jnp.float32) / 255.0

def visualize_prediction(image, pred_class, true_class):
    plt.imshow(image.astype(np.float32))
    plt.title(f"Prediction: {classes[pred_class]}\nGround Truth: {classes[true_class]}")
    plt.axis('off')
    plt.show()

def run_multiple_test_samples(model, params, test_data, num_samples=20):
    print(f"\n📷 Running inference on {num_samples} random test images...")
    correct = 0
    error_stats = {}

    for _ in range(num_samples):
        imgs, labels = test_data[np.random.randint(len(test_data))]
        image = imgs[0]
        label = int(labels[0])

        image_norm = image[None, ...]
        logits = model.apply({'params': params}, image_norm)
        pred_class = int(jnp.argmax(logits, axis=-1)[0])

        visualize_prediction(image, pred_class, label)
        print(f"🔹 Predicted: {classes[pred_class]}")
        print(f"🔸 Ground Truth: {classes[label]}\n")

        if pred_class == label:
            correct += 1
        else:
            true_label = classes[label]
            pred_label = classes[pred_class]
            error_stats.setdefault((true_label, pred_label), 0)
            error_stats[(true_label, pred_label)] += 1

    acc = correct / num_samples
    print(f"✅ Correct Predictions: {correct}/{num_samples}")
    print(f"📈 Accuracy: {acc:.2f}")

def main():
    total_start = time.time()  # ⏱️ 整體開始時間
    rng = jax.random.PRNGKey(0)

    model = MlpMixer(
        num_classes=10,
        num_blocks=4,
        patch_size=7,
        hidden_dim=64,
        tokens_mlp_dim=128,
        channels_mlp_dim=256,
    )

    num_epochs = 1
    batch_size = 64
    learning_rate = 5e-3
    warmup_steps = 200

    train_data = load_dataset(train=True, batch_size=batch_size)
    test_data = load_dataset(train=False, batch_size=batch_size)
    total_steps = num_epochs * len(train_data)

    # ✅ 建立 learning rate schedule
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=total_steps - warmup_steps,
        end_value=0.0,
    )

    state = create_train_state(
        rng,
        model,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        total_steps=total_steps
    )

    accs, losses, lrs = [], [], []
    for epoch in range(num_epochs):
        start_time = time.time()  # ⏳ 開始計時
        for batch in train_data:
            state, metrics = train_step(state, batch)
            
        end_time = time.time()  # ⏱️ 結束計時
        elapsed = end_time - start_time

        current_step = epoch * len(train_data)
        current_lr = lr_schedule(current_step)

        accs.append(metrics['accuracy'])
        losses.append(metrics['loss'])
        lrs.append(current_lr)

        print(f"📉 Epoch {epoch+1} — LR: {current_lr:.8f}, Loss: {metrics['loss']:.6f}, Acc: {metrics['accuracy']:.4f} , time: {elapsed:.2f} 秒")
    
    test_acc = evaluate(model, state.params, test_data)
    print(f"\n✅ Test Accuracy: {test_acc:.4f}")
    
    total_end = time.time()  # ⏱️ 整體結束時間
    total_elapsed = total_end - total_start
    print(f"🕒 總執行時間：{total_elapsed:.2f} 秒")
    
    plot_metrics(losses)
    plot_metrics(accs)
    plot_metrics(lrs)

    print("\n🔍 Multiple image inference from test set:")
    run_multiple_test_samples(model, state.params, test_data, num_samples=5) #抽樣

if __name__ == "__main__":
    main()