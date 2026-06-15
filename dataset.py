# =================================================================================
# Chargement et tokenization des tweets pour la classification de sentiment
# =================================================================================

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import pandas as pd
from sklearn.model_selection import train_test_split

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_NAME = "google-bert/bert-base-uncased"
MAX_LENGTH = 128

# Mapping des classes en indices
LABEL2ID = {
    "Positive"          : 0,
    "Negative"          : 1,
    "Neutral"           : 2,
    "Extremely Positive": 3,
    "Extremely Negative": 4
}

ID2LABEL = {v: k for k, v in LABEL2ID.items()}


# ============================================================
# CLASSE TEXTCLASSIFICATIONDATASET
# ============================================================
class TextClassificationDataset(Dataset):

    def __init__(self, texts, labels, tokenizer, max_length=MAX_LENGTH):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):

        text  = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length     = self.max_length,
            padding        = "max_length",
            truncation     = True,
            return_tensors = "pt"
        )

        return {
            "input_ids"     : encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label"         : torch.tensor(label, dtype=torch.long)
        }


# ============================================================
# FONCTION GET_DATALOADERS
# ============================================================
def get_dataloaders(data_path: str, batch_size: int,
                    num_workers: int = 0):

    # Charger le dataset
    df = pd.read_csv(data_path, encoding='latin-1')

    # Garder seulement les colonnes utiles
    df = df[['OriginalTweet', 'Sentiment']].dropna()

    # Convertir les labels en indices
    df['label'] = df['Sentiment'].map(LABEL2ID)

    # Split train/val 80/20 stratifie
    train_df, val_df = train_test_split(
        df,
        test_size    = 0.2,
        random_state = 42,
        stratify     = df['label']
    )

    # Charger le tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Creer les datasets
    train_dataset = TextClassificationDataset(
        texts     = train_df['OriginalTweet'].values,
        labels    = train_df['label'].values,
        tokenizer = tokenizer
    )

    val_dataset = TextClassificationDataset(
        texts     = val_df['OriginalTweet'].values,
        labels    = val_df['label'].values,
        tokenizer = tokenizer
    )

    # Creer les DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers
    )

    # Affichage des informations
    print(f"\n=== DATASET ===")
    print(f"Total         : {len(df)} tweets")
    print(f"Train         : {len(train_dataset)} tweets")
    print(f"Val           : {len(val_dataset)} tweets")
    print(f"Classes       : {list(LABEL2ID.keys())}")
    print(f"Max length    : {MAX_LENGTH} tokens")

    return train_loader, val_loader, tokenizer


# ============================================================
# TEST DU DATASET
# ============================================================
if __name__ == "__main__":
    train_loader, val_loader, tokenizer = get_dataloaders(
        data_path  = "data/Corona_NLP_train.csv",
        batch_size = 32
    )

    # Verifier un batch
    batch = next(iter(train_loader))
    print(f"\nInput IDs shape      : {batch['input_ids'].shape}")
    print(f"Attention mask shape : {batch['attention_mask'].shape}")
    print(f"Labels shape         : {batch['label'].shape}")