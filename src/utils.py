from __future__ import annotations

import json
import os
import pickle
from typing import Any

import pandas as pd
import s3fs
import torch
from prettytable import PrettyTable
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split


def _s3() -> s3fs.S3FileSystem:
    return s3fs.S3FileSystem(
        anon=False,
        key=os.environ["AWS_ACCESS_KEY_ID"],
        secret=os.environ["AWS_SECRET_ACCESS_KEY"],
        client_kwargs={'endpoint_url': os.environ["AWS_ENDPOINT_URL"]})


def write_pickle_to_s3(obj: Any, s3_path: str) -> None:
    s3 = _s3()
    pickle.dump(obj, s3.open(s3_path, 'wb'))


def read_pickle_from_s3(s3_path: str) -> Any:
    s3 = _s3()
    with s3.open(s3_path, 'rb') as f:
        data = json.load(f)
    return data


def read_parquet_s3(s3_path: str) -> pd.DataFrame:
    options = {
        'client_kwargs': {
            'endpoint_url': os.environ["AWS_ENDPOINT_URL"],
        }
    }

    return pd.read_parquet(s3_path, storage_options=options)


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def prepare_train_test_split_data(data: pd.DataFrame):
    train, val = train_test_split(data, test_size=0.1, stratify=data["sentiment"].tolist())
    val, test = train_test_split(val, test_size=0.2, stratify=val["sentiment"].tolist())

    options = {
        'client_kwargs': {
            'endpoint_url': os.environ["AWS_ENDPOINT_URL"],
        }
    }

    val.to_parquet(os.environ["VAL_PATH"], storage_options=options)
    train.to_parquet(os.environ["TRAIN_PATH"], storage_options=options)
    test.to_parquet(os.environ["TEST_PATH"], storage_options=options)


class Metric:
    def __init__(self,
                 mlflow: Any,
                 len_train_dataset: int | None = None,
                 len_val_dataset: int | None = None,
                 len_test_dataset: int | None = None, ):
        self.mlflow = mlflow
        self.metric_threshold = 0.5

        self.y = []
        self.t = []
        self.loss = 0
        self.step = "train"

        self.last_val_loss = 0
        self.last_train_loss = 0

        self.best_val_loss = 0
        self.best_train_loss = 0

        self.len_train_dataset = len_train_dataset
        self.len_val_dataset = len_val_dataset
        self.len_test_dataset = len_test_dataset

        if len_train_dataset:
            self.mlflow.log_param("len_train_dataset", len_train_dataset)
        if len_val_dataset:
            self.mlflow.log_param("len_val_dataset", len_val_dataset)
        if len_test_dataset:
            self.mlflow.log_param("len_test_dataset", len_test_dataset)

    def reset(self, step: str):
        self.y = []
        self.t = []
        self.loss = 0
        self.step = step

    def update(self, y, t, loss):
        """Update with batch outputs and labels.

        Args:
          y: (tensor) model outputs sized [N,].
          t: (tensor) labels targets sized [N,].
          loss: (int).
        """
        self.y.append(y.argmax(-1).cpu())
        self.t.append(t.cpu())
        self.loss += loss.cpu().item()

    def compute(self):
        precision = self.precision()
        recall = self.recall()
        loss = self.compute_loss()

        self.mlflow.log_metric(f"{self.step}_loss", loss)
        self.mlflow.log_metric(f"{self.step}_precision", precision)
        self.mlflow.log_metric(f"{self.step}_recall", recall)

        if self.step == "train":
            self.last_train_loss = loss
        elif self.step == "val":
            self.last_val_loss = loss

    def precision(self):
        y = torch.cat(self.y, 0)
        t = torch.cat(self.t, 0)
        return precision_score(t, y, zero_division=0.0, average="micro")

    def recall(self):
        y = torch.cat(self.y, 0)
        t = torch.cat(self.t, 0)
        return recall_score(t, y, zero_division=0.0, average="micro")

    def compute_loss(self):
        if self.step == "train":
            return self.loss / self.len_train_dataset
        elif self.step == "val":
            return self.loss / self.len_val_dataset
        elif self.step == "test":
            return self.loss / self.len_test_dataset
        else:
            raise NameError(f"train or val, but {self.step} was given ...")
