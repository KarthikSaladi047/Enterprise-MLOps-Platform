"""
Katib trial training script.

Reads a dataset CSV (columns: text, label) from DATASET_PATH, splits it into
train/val, fine-tunes a small text classifier, and prints the validation loss
in `final_loss=<value>` format for Katib's StdOut metrics collector.

Expected environment variables:
  TRIAL_LEARNING_RATE  - learning rate (float), required
  TRIAL_BATCH_SIZE     - batch size (int), required
  DATASET_PATH         - path to dataset CSV (default: /shared/dataset.csv)
  MODEL_NAME           - HF model id (default: prajjwal1/bert-tiny)
  EPOCHS               - number of training epochs (default: 3)
"""
import os
import warnings

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import logging as hf_logging

warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")
hf_logging.set_verbosity_error()


def make_loader(tokenizer, df, batch_size, shuffle):
    enc = tokenizer(
        df["text"].tolist(),
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    labels = torch.tensor(df["label"].tolist())
    dataset = TensorDataset(enc["input_ids"], enc["attention_mask"], labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def main():
    learning_rate = float(os.environ["TRIAL_LEARNING_RATE"])
    batch_size = int(os.environ["TRIAL_BATCH_SIZE"])
    dataset_path = os.environ.get("DATASET_PATH", "/shared/dataset.csv")
    model_name = os.environ.get("MODEL_NAME", "prajjwal1/bert-tiny")
    epochs = int(os.environ.get("EPOCHS", "3"))

    df = pd.read_csv(dataset_path)
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label"] if df["label"].nunique() > 1 else None,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    train_loader = make_loader(tokenizer, train_df, batch_size, shuffle=True)
    val_loader = make_loader(tokenizer, val_df, batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        for b_ids, b_mask, b_labels in train_loader:
            optimizer.zero_grad()
            out = model(input_ids=b_ids, attention_mask=b_mask, labels=b_labels)
            out.loss.backward()
            optimizer.step()

    model.eval()
    val_loss_total = 0.0
    with torch.no_grad():
        for b_ids, b_mask, b_labels in val_loader:
            out = model(input_ids=b_ids, attention_mask=b_mask, labels=b_labels)
            val_loss_total += out.loss.item()
    val_loss = val_loss_total / len(val_loader)

    # Picked up by Katib's StdOut metrics collector (name=value format)
    print(f"final_loss={val_loss}")


if __name__ == "__main__":
    main()