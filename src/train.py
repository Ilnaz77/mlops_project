from __future__ import annotations

import os

import mlflow
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import Collate, QueryDataset, VocabularyWords
from model import RNNModel
from utils import Metric


def main(
        batch_size: int = 32,
        num_epochs: int = 100,
        embed_size: int = 100,
        hidden_size: int = 100,
        max_grad_norm: int = 7,
        n_freq: int = 5,
):
    with mlflow.start_run():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        mlflow.log_param("device", device.type)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("embed_size", embed_size)
        mlflow.log_param("hidden_size", hidden_size)
        mlflow.log_param("n_freq", n_freq)

        vocab_words = VocabularyWords(n_freq=n_freq)

        train_dataset = QueryDataset(file_path=os.environ["TRAIN_PATH"], vocab_words=vocab_words)
        val_dataset = QueryDataset(file_path=os.environ["VAL_PATH"], vocab_words=vocab_words)

        vocab_words.build_vocabulary(train_dataset.data.text.tolist())
        pad_idx = vocab_words.stoi["<PAD>"]

        mlflow.log_param("vocab_word_size", len(vocab_words))
        vocab_words.save_vocab(mlflow)

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            collate_fn=Collate(pad_idx=pad_idx),
            shuffle=True,
        )

        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            collate_fn=Collate(pad_idx=pad_idx),
            shuffle=False,
        )

        model = RNNModel(
            vocab_size=len(vocab_words),
            output_size=3,
            embed_size=embed_size,
            hidden_size=hidden_size,
            pad_idx=pad_idx,
        ).to(device)

        mlflow.log_param("pad_idx", pad_idx)
        mlflow.log_param("output_size", 3)

        criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, reduction="sum")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        metric = Metric(len_train_dataset=len(train_dataset), len_val_dataset=len(val_dataset), mlflow=mlflow)

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=10,
            min_lr=0.0001,
        )

        best_val_loss = 1e34

        with tqdm(range(num_epochs), unit="epoch", position=0, leave=True) as tepoch:
            for epoch in tepoch:
                model.train()
                metric.reset(step="train")
                for batch in train_loader:
                    optimizer.zero_grad()
                    words, labels, seq_lengths = (
                        batch["words"].to(device),
                        batch["labels"].to(device),
                        batch["seq_lengths"],
                    )
                    predictions = model(words, seq_lengths)
                    loss = criterion(predictions, labels)
                    metric.update(predictions, labels, loss)
                    loss.backward()
                    if max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                metric.compute()

                model.eval()
                metric.reset(step="val")
                with torch.no_grad():
                    for batch in val_loader:
                        words, labels, seq_lengths = (
                            batch["words"].to(device),
                            batch["labels"].to(device),
                            batch["seq_lengths"],
                        )
                        predictions = model(words, seq_lengths)
                        loss = criterion(predictions, labels)
                        metric.update(predictions, labels, loss)
                metric.compute()

                scheduler.step(metric.last_val_loss)

                if metric.last_val_loss < best_val_loss:
                    mlflow.log_metric("best_val_loss", metric.last_val_loss)
                    best_val_loss = metric.last_val_loss
                    mlflow.pytorch.log_state_dict(model.state_dict(), os.environ["MODEL_ARTIFACT_PATH"])

                tepoch.set_postfix(epoch=epoch + 1, loss_train=metric.last_train_loss,
                                   loss_val=metric.last_val_loss, curr_lr=optimizer.param_groups[0]["lr"])


def run_train(kwargs: dict):
    mlflow.set_tracking_uri(f"http://{os.environ['TRACKING_SERVER_HOST']}:5000")
    mlflow.set_experiment(os.environ["MLFLOW_TRAIN_EXPERIMENT_NAME"])

    main(**kwargs)


if __name__ == "__main__":
    kwargs = {
        "batch_size": 256,
        "num_epochs": 3,
        "embed_size": 5,
        "hidden_size": 5,
        "n_freq": 1,
    }
    run_train(kwargs)
