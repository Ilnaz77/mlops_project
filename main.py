from __future__ import annotations

import hashlib
import os
from typing import Tuple, Any

import pandas as pd
from loguru import logger
from mlflow import MlflowClient
from mlflow.exceptions import RestException
from prefect import task, flow

from src.register_model import update_production_model
from src.train import run_train
from src.utils import read_parquet_s3, prepare_train_test_split_data


@task(retries=2, retry_delay_seconds=15, name="Check data has changed or not")
def check_data() -> Tuple[pd.DataFrame, bool, Any]:
    """
    For simplicity and not leaking data, we will choose best model within one experiment.

    1) First scenario. Let's assume dataset with texts can be updated once a week or once a month.
    For example, DWH team add new "feedbacks" from online-retail shop as "deliveroo" etc.
    2) Second scenario. Let's assume data is not updated by new lines, but some ML engineer decided to remove stop words
    from origin data. So, we have new dataset, and we should re-train model.

    So we want to check data is the same or is changed. If data has changed, we run:
        - train.py on this updated data
        - register_model.py to check our new data is good or not in test dataset
    """
    old_data = read_parquet_s3(os.environ["OLD_DATA"])
    curr_data = read_parquet_s3(os.environ["CURR_DATA"])  # may be changed by DWH team or by ML engineer

    hash_old = hashlib.sha256(old_data.to_json().encode()).hexdigest()
    hash_new = hashlib.sha256(curr_data.to_json().encode()).hexdigest()

    return curr_data, hash_old == hash_new, hash_new


@task(name="Check model")
def check_model() -> bool:
    """
    We must avoid the situation, when there is no Production model.
    """
    client = MlflowClient(tracking_uri=f"http://{os.environ['TRACKING_SERVER_HOST']}:5000")
    try:
        latest_versions = client.get_latest_versions(name=os.environ["MODEL_NAME"])
    except RestException:
        return False  # there is no any model

    check = False
    for version in latest_versions:
        if version.current_stage == "Production":
            check = True
    return check


@task(retries=1, name="Train model if data is updated")
def train_model():
    """
    There we save bests possible configs of model.
    """
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


@task(name="New data split to train/test/val")
def split_data(curr_data: pd.DataFrame):
    """
    New data should be splitted.
    """
    prepare_train_test_split_data(curr_data)


@task(name="Update production model")
def update_prod_model():
    """
    Best model, which has the little loss in test dataset, will become Production.
    """
    update_production_model()


@flow(name="Main workflow")
def main():
    data, data_check, data_hash = check_data()
    model_check = check_model()

    if model_check and data_check:
        logger.info("Data haven't changed and Production model is ready ...")
        return None

    os.environ["MLFLOW_EXPERIMENT_NAME"] = data_hash

    split_data(data)
    train_model()
    update_prod_model()


if __name__ == "__main__":
    main()
