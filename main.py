# ============================================================
# Pipeline principal Inspection + Entrainement + Evaluation
# ============================================================

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import set_seed, plot_training_curves, plot_confusion_matrix
from dataset import get_dataloaders, LABEL2ID, ID2LABEL
from model import get_model
from train import train_epoch, eval_epoch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

# ============================================================
# CONFIGURATION
# ============================================================
CONFIG = {
    "data_path"    : "data/Corona_NLP_train.csv",
    "model_path"   : "best_model.pth",
    "num_classes"  : 5,
    "batch_size"   : 16,
    "lr"           : 2e-5,
    "epochs"       : 3,
    "max_length"   : 128,
    "dropout"      : 0.3,
    "weight_decay" : 0.01,
    "warmup_ratio" : 0.1,
    "seed"         : 42,
}

# ============================================================
# INSPECTION DU DATASET
# ============================================================
def inspect_dataset(data_path: str):

    print("\n" + "="*50)
    print("INSPECTION DU DATASET")
    print("="*50)

    df = pd.read_csv(data_path, encoding='latin-1')

    print(f"\nTotal tweets : {len(df)}")
    print(f"Colonnes     : {df.columns.tolist()}")

    print("\n--- DISTRIBUTION DES CLASSES ---")
    print(df['Sentiment'].value_counts())

    ratio = df['Sentiment'].value_counts().max() / \
            df['Sentiment'].value_counts().min()
    print(f"\nRatio max/min : {ratio:.2f}")
    if ratio > 2:
        print("Desequilibre detecte ! Strategie : weighted loss")
    else:
        print("Dataset acceptable")

    print("\n--- LONGUEUR DES TWEETS ---")
    longueurs = df['OriginalTweet'].str.split().str.len()
    print(f"Min     : {longueurs.min()}")
    print(f"Max     : {longueurs.max()}")
    print(f"Moyenne : {longueurs.mean():.0f}")
    print(f"Max length choisi : {CONFIG['max_length']} tokens")

    print("\n--- 5 EXEMPLES ---")
    for i, row in df.sample(5, random_state=42).iterrows():
        print(f"\nTweet     : {row['OriginalTweet'][:100]}...")
        print(f"Sentiment : {row['Sentiment']}")


# ============================================================
# FONCTION PRINCIPALE
# ============================================================
def main():

    # Fixer la seed
    set_seed(CONFIG["seed"])

    # Device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"\nDevice : {device}")

    # Inspection du dataset
    inspect_dataset(CONFIG["data_path"])

    # Dataset
    train_loader, val_loader, tokenizer = get_dataloaders(
        data_path  = CONFIG["data_path"],
        batch_size = CONFIG["batch_size"]
    )

    # Modele
    model = get_model(
        num_classes = CONFIG["num_classes"],
        dropout     = CONFIG["dropout"]
    ).to(device)

    # Loss optimizer scheduler
    criterion    = nn.CrossEntropyLoss()
    optimizer    = AdamW(
        model.parameters(),
        lr           = CONFIG["lr"],
        weight_decay = CONFIG["weight_decay"]
    )

    total_steps  = len(train_loader) * CONFIG["epochs"]
    warmup_steps = int(total_steps * CONFIG["warmup_ratio"])
    scheduler    = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = warmup_steps,
        num_training_steps = total_steps
    )

    # Historique
    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []
    best_val_loss = float("inf")

    print(f"\n=== ENTRAINEMENT BERT ===")

    for epoch in range(1, CONFIG["epochs"] + 1):

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer,
            criterion, scheduler, device
        )

        val_loss, val_acc, val_f1 = eval_epoch(
            model, val_loader, criterion, device
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(
            f"Epoch {epoch}/{CONFIG['epochs']} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"F1: {val_f1:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(),
                      CONFIG["model_path"])
            print(f"Meilleur modele sauvegarde !")

    # Courbes d entrainement
    plot_training_curves(
        train_losses, val_losses,
        train_accs,   val_accs
    )

    # Matrice de confusion
    print("\n=== EVALUATION FINALE ===")
    model.load_state_dict(torch.load(CONFIG["model_path"]))
    model.eval()

    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['label'].to(device)

            logits = model(input_ids, attention_mask)
            preds  = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    class_names = list(LABEL2ID.keys())
    plot_confusion_matrix(all_preds, all_labels, class_names)

    print("\nTOUS LES ENTRAINEMENTS TERMINES !")


# ============================================================
# POINT D ENTREE
# ============================================================
if __name__ == "__main__":
    main()