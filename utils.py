import matplotlib.pyplot as plt
import numpy as np

def plot_all_metrics(train_accs, train_losses, val_accs, val_losses, lrs):
    epochs = range(1, len(train_accs) + 1)
    fig, ax1 = plt.subplots(figsize=(10,6))

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy', color='tab:blue')
    l1, = ax1.plot(epochs, train_accs, label='Train Accuracy', color='tab:blue', linestyle='-')
    l2, = ax1.plot(epochs, val_accs, label='Val Accuracy', color='tab:blue', linestyle='--')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Loss', color='tab:red')
    l3, = ax2.plot(epochs, train_losses, label='Train Loss', color='tab:red', linestyle='-')
    l4, = ax2.plot(epochs, val_losses, label='Val Loss', color='tab:red', linestyle='--')
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
    
def plot_metrics(lrs, title="Learning_Rate"):
    plt.figure(figsize=(10, 6))
    plt.plot(lrs)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.grid(True)
    plt.show()