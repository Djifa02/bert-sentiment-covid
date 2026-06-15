# ============================================================
# Fonctions utilitaires metriques, seed, visualisations
# ============================================================

import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


# ============================================================
# FONCTION SET SEED
# ============================================================
def set_seed(seed: int = 42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"Seed fixee : {seed}")


# ============================================================
# FONCTION PLOT TRAINING CURVES
# ============================================================
def plot_training_curves(train_losses, val_losses,
                         train_accs, val_accs):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(train_losses) + 1)

    # Courbe de loss
    ax1.plot(epochs, train_losses, label='Train Loss')
    ax1.plot(epochs, val_losses,   label='Val Loss')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Courbe d accuracy
    ax2.plot(epochs, train_accs, label='Train Accuracy')
    ax2.plot(epochs, val_accs,   label='Val Accuracy')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    fig.tight_layout()
    plt.savefig("training_curves.png")
    plt.close()
    print("Courbes sauvegardees : training_curves.png")


# ============================================================
# FONCTION PLOT CONFUSION MATRIX
# ============================================================
def plot_confusion_matrix(preds, labels, class_names):

    cm = confusion_matrix(labels, preds)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot      = True,
        fmt        = 'd',
        cmap       = 'Blues',
        xticklabels = class_names,
        yticklabels = class_names
    )
    ax.set_title('Matrice de Confusion')
    ax.set_ylabel('Vrai label')
    ax.set_xlabel('Label predit')

    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()
    print("Matrice sauvegardee : confusion_matrix.png")

    print("\n=== RAPPORT DE CLASSIFICATION ===")
    print(classification_report(
        labels, preds,
        target_names=class_names
    ))


# ============================================================
# TEST
# ============================================================
if __name__ == "__main__":
    set_seed(42)
    print("utils.py OK !")