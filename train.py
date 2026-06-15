# =============================================================================
# Boucles d entrainement et d evaluation + fonction principale main
# =============================================================================

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
from tqdm import tqdm


# ============================================================
# FONCTION TRAIN EPOCH
# ============================================================
def train_epoch(model, loader, optimizer,
                criterion, scheduler, device):

    model.train()
    total_loss    = 0.0
    total_correct = 0
    total_samples = 0

    loop = tqdm(loader, leave=False)

    for batch in loop:

        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels         = batch['label'].to(device)

        # Remettre les gradients a zero
        optimizer.zero_grad()

        # Forward pass
        logits = model(input_ids, attention_mask)

        # Calcul de la loss
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0
        )

        # Mise a jour des poids
        optimizer.step()

        # Mise a jour du scheduler
        scheduler.step()

        # Accumulation des metriques
        total_loss    += loss.item() * input_ids.size(0)
        total_samples += input_ids.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()

        loop.set_postfix(loss=loss.item())

    avg_loss     = total_loss    / total_samples
    avg_accuracy = total_correct / total_samples

    return avg_loss, avg_accuracy


# ============================================================
# FONCTION EVAL EPOCH
# ============================================================
def eval_epoch(model, loader, criterion, device):

    model.eval()
    total_loss    = 0.0
    total_correct = 0
    total_samples = 0
    all_preds     = []
    all_labels    = []

    with torch.no_grad():
        loop = tqdm(loader, leave=False)

        for batch in loop:

            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['label'].to(device)

            # Forward pass uniquement
            logits = model(input_ids, attention_mask)
            loss   = criterion(logits, labels)

            total_loss    += loss.item() * input_ids.size(0)
            total_samples += input_ids.size(0)

            preds          = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            loop.set_postfix(loss=loss.item())

    avg_loss     = total_loss    / total_samples
    avg_accuracy = total_correct / total_samples
    f1           = f1_score(all_labels, all_preds,
                            average='weighted')

    return avg_loss, avg_accuracy, f1


# ============================================================
# FONCTION MAIN
# ============================================================
def main():

    import random
    import numpy as np
    from dataset import get_dataloaders
    from model import get_model

    # ========================================================
    # CONFIGURATION
    # ========================================================
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

    # ========================================================
    # FIXER LA SEED
    # ========================================================
    random.seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])
    torch.cuda.manual_seed_all(CONFIG["seed"])

    # ========================================================
    # DEVICE
    # ========================================================
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device : {device}")

    # ========================================================
    # DATASET
    # ========================================================
    train_loader, val_loader, tokenizer = get_dataloaders(
        data_path  = CONFIG["data_path"],
        batch_size = CONFIG["batch_size"]
    )

    # ========================================================
    # MODELE
    # ========================================================
    model = get_model(
        num_classes = CONFIG["num_classes"],
        dropout     = CONFIG["dropout"]
    ).to(device)

    # ========================================================
    # LOSS OPTIMIZER SCHEDULER
    # ========================================================
    criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(
        model.parameters(),
        lr           = CONFIG["lr"],
        weight_decay = CONFIG["weight_decay"]
    )

    total_steps  = len(train_loader) * CONFIG["epochs"]
    warmup_steps = int(total_steps * CONFIG["warmup_ratio"])

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = warmup_steps,
        num_training_steps = total_steps
    )

    # ========================================================
    # ENTRAINEMENT
    # ========================================================
    best_val_loss = float("inf")

    print(f"\n=== ENTRAINEMENT BERT ===")
    print(f"Epochs     : {CONFIG['epochs']}")
    print(f"Batch size : {CONFIG['batch_size']}")
    print(f"LR         : {CONFIG['lr']}")

    for epoch in range(1, CONFIG["epochs"] + 1):

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer,
            criterion, scheduler, device
        )

        val_loss, val_acc, val_f1 = eval_epoch(
            model, val_loader, criterion, device
        )

        print(
            f"Epoch {epoch}/{CONFIG['epochs']} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"F1: {val_f1:.4f}"
        )

        # Sauvegarde du meilleur modele
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(),
                      CONFIG["model_path"])
            print(f"Meilleur modele sauvegarde !")

    print("\nENTRAINEMENT TERMINE !")
    print(f"Meilleur Val Loss : {best_val_loss:.4f}")


# ============================================================
# POINT D ENTREE
# ============================================================
if __name__ == "__main__":
    main()