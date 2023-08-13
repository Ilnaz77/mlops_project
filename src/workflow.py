from __future__ import annotations

import hashlib
import os
from datetime import datetime

import pandas as pd
from loguru import logger
from prefect import task, flow

from register_model import update_production_model
from train import run_train
from utils import read_parquet_s3, prepare_train_test_split_data


@task(retries=2, retry_delay_seconds=15, name="Check data has changed or not")
def check_data() -> pd.DataFrame | None:
    """
    Let's assume dataset with texts can be updated once a week or once a month. For example, DWH team add new
    "feedbacks" from online-retail shop as "deliveroo" etc.
    So we want to check data is the same or is changed. If data has changed, we run:
        - train.py on this updated data
        - register_model.py to check our new data is good or not in test dataset
    :return:
    """
    old_data = read_parquet_s3(os.environ["OLD_DATA"])
    curr_data = read_parquet_s3(os.environ["CURR_DATA"])  # may be changed by DWH team

    hash_old = hashlib.sha256(old_data.to_json().encode()).hexdigest()
    hash_new = hashlib.sha256(curr_data.to_json().encode()).hexdigest()

    if hash_old != hash_new:
        return curr_data
    else:
        return None


@task(name="New data split to train/test/val")
def split_data(curr_data: pd.DataFrame):
    prepare_train_test_split_data(curr_data)


@task(retries=1, name="Train model if data is updated")
def train_model():
    run_train({
        "batch_size": 256,
        "num_epochs": 3,
        "embed_size": 5,
        "hidden_size": 5,
        "n_freq": 1,
    })

    run_train({
        "batch_size": 256,
        "num_epochs": 3,
        "embed_size": 10,
        "hidden_size": 10,
        "n_freq": 1,
    })


@task(name="Update production model")
def update_prod_model():
    update_production_model()


@flow(name="Main workflow")
def main():
    data = check_data()
    if data is None:
        logger.info("Data haven't changed ...")
        return None

    os.environ[
        "MLFLOW_TRAIN_EXPERIMENT_NAME"] = f"{os.environ['MLFLOW_TRAIN_EXPERIMENT_NAME']}_{datetime.today().strftime('%Y-%m-%d')}"

    split_data()
    train_model()
    update_prod_model()


if __name__ == "__main__":
    main()