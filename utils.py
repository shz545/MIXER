import matplotlib.pyplot as plt
import numpy as np

def cutmix_data(x, y, alpha=1.0):
    """CutMix augmentation"""
    batch_size, h, w, c = x.shape
    lam = np.random.beta(alpha, alpha)
    rand_index = np.random.permutation(batch_size)
    y_a = y
    y_b = y[rand_index]

    # 隨機產生 bbox
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int32(w * cut_rat)
    cut_h = np.int32(h * cut_rat)

    cx = np.random.randint(w)
    cy = np.random.randint(h)
    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)

    x_cutmix = np.copy(x)
    for i in range(batch_size):
        x_cutmix[i, bby1:bby2, bbx1:bbx2, :] = x[rand_index[i], bby1:bby2, bbx1:bbx2, :]

    # 修正 lam
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
    return x_cutmix, y_a, y_b, lam

def plot_all_metrics(train_accs, train_losses, Val_accs, Val_losses, lrs):
    epochs = range(1, len(train_accs) + 1)
    fig, ax1 = plt.subplots(figsize=(10,6))

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy', color='tab:blue')
    l1, = ax1.plot(epochs, train_accs, label='Train Accuracy', color='tab:blue', linestyle='-')
    l2, = ax1.plot(epochs, Val_accs, label='Val Accuracy', color='tab:blue', linestyle='--')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Loss', color='tab:red')
    l3, = ax2.plot(epochs, train_losses, label='Train Loss', color='tab:red', linestyle='-')
    l4, = ax2.plot(epochs, Val_losses, label='Val Loss', color='tab:red', linestyle='--')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    ax3.set_ylabel('Learning Rate', color='tab:green')
    l5, = ax3.plot(np.linspace(1, len(train_accs), len(lrs)), lrs, label='Learning Rate', color='tab:green', alpha=0.3)
    ax3.tick_params(axis='y', labelcolor='tab:green')

    lines = [l1, l2, l3, l4, l5]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)

    plt.title('Training/Val Accuracy, Loss and Learning Rate')
    fig.tight_layout()
    plt.show()