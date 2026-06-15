# =================================================================================
# Chargement et definition du modele BERT pour la classification de sentiment
# =================================================================================

import torch
import torch.nn as nn
from transformers import BertModel, AutoTokenizer
from dataset import MODEL_NAME, ID2LABEL, LABEL2ID

# ============================================================
# CLASSE BERTCLASSIFIER
# ============================================================
class BertClassifier(nn.Module):

    def __init__(self, num_classes: int = 5,
                 dropout: float = 0.3,
                 freeze_bert: bool = False):

        super(BertClassifier, self).__init__()

        # Charger BERT pre-entraine
        self.bert = BertModel.from_pretrained(MODEL_NAME)

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):

        outputs = self.bert(
            input_ids      = input_ids,
            attention_mask = attention_mask
        )

        cls_output = outputs.last_hidden_state[:, 0, :]

        # Appliquer le dropout
        cls_output = self.dropout(cls_output)

        # Tete de classification
        logits = self.classifier(cls_output)

        return logits


# ============================================================
# FONCTION GET_MODEL
# ============================================================
def get_model(num_classes: int = 5,
              dropout: float = 0.3,
              freeze_bert: bool = False):

    model = BertClassifier(
        num_classes = num_classes,
        dropout     = dropout,
        freeze_bert = freeze_bert
    )

    # Affichage des informations
    total_params    = sum(p.numel()
                         for p in model.parameters())
    trainable_params = sum(p.numel()
                          for p in model.parameters()
                          if p.requires_grad)

    print(f"\n=== MODELE BERT ===")
    print(f"Modele           : {MODEL_NAME}")
    print(f"Classes          : {num_classes}")
    print(f"Total params     : {total_params:,}")
    print(f"Params entraines : {trainable_params:,}")

    return model


# ============================================================
# TEST DU MODELE
# ============================================================
if __name__ == "__main__":
    import torch

    model = get_model()

    # Creer des donnees fictives pour tester
    batch_size   = 4
    max_length   = 128
    input_ids    = torch.randint(0, 1000, (batch_size, max_length))
    attention_mask = torch.ones(batch_size, max_length)

    # Passer dans le modele
    logits = model(input_ids, attention_mask)

    print(f"\nInput shape  : {input_ids.shape}")
    print(f"Output shape : {logits.shape}")
    print("model.py OK !")