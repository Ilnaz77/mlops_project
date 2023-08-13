import os
from typing import Tuple

import mlflow
import torch
from mlflow import MlflowClient

from src.dataloader import VocabularyWords
from src.model import load_model_from_s3


def get_sentiment(text: str, model, tokenizer, device) -> str:
    mapping = {0: "negative", 1: "neutral", 2: "positive"}
    tokens = torch.LongTensor(tokenizer.encode(text).ids).unsqueeze(0).to(device)
    sentiment = model.inference(tokens).argmax(-1).cpu().item()
    return mapping[sentiment]


def get_prod_model() -> Tuple:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mlflow.set_tracking_uri(f"http://{os.environ['TRACKING_SERVER_HOST']}:5000")
    client = MlflowClient()

    exp_id = client.get_experiment_by_name(os.environ["MLFLOW_TRAIN_EXPERIMENT_NAME"]).experiment_id
    latest_versions = client.get_latest_versions(name=os.environ["MODEL_NAME"])

    run_id = None
    for version in latest_versions:
        if version.current_stage == "Production":
            run_id = version.run_id
    if run_id is None:
        raise Exception("There is not production model ...")

    tokenizer = _get_tokenizer(run_id, exp_id)
    model = load_model_from_s3(exp_id, run_id, device)

    return model, tokenizer, device


def _get_tokenizer(run_id: str, exp_id: int):
    vocab = VocabularyWords()
    vocab.read_vocab(run_id, exp_id)
    return vocab.tokenizer


if __name__ == "__main__":
    model, tokenizer, device = get_prod_model()
    get_sentiment("i like this film", model, tokenizer, device)
