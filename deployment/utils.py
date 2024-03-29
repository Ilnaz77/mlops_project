import os
from typing import Tuple

import torch
from mlflow import MlflowClient
from tokenizers import Tokenizer
from mlflow.exceptions import RestException

from src.model import RNNModel, load_model_from_s3
from src.utils import clean_text
from src.dataloader import VocabularyWords


def get_sentiment(text: str, model: RNNModel, tokenizer: Tokenizer, device: torch.device) -> str:
    if not isinstance(text, str):
        raise Exception(f"text should be str, but {type(str)} was given ...")

    text = clean_text(text)

    if text == "":
        return "There is empty sentence after normalization ..."

    mapping = {
        0: "negative",
        1: "neutral",
        2: "positive",
    }
    tokens = torch.LongTensor(tokenizer.encode(text).ids).unsqueeze(0).to(device)
    sentiment = model.inference(tokens).argmax(-1).cpu().item()
    return mapping[sentiment]


def get_prod_model() -> Tuple:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    client = MlflowClient(tracking_uri=f"http://{os.environ['TRACKING_SERVER_HOST']}:5000")

    try:
        latest_versions = client.get_latest_versions(name=os.environ["MODEL_NAME"])
    except RestException:
        raise RestException("There is no any model! Try to train new one via running main.py ...")

    run_id, exp_id = None, None
    for version in latest_versions:
        if version.current_stage == "Production":
            run_id = version.run_id
            exp_id = version.tags["exp_id"]
    if run_id is None or exp_id is None:
        raise Exception(f"There is not production model ... run_id is {run_id}, exp_id is {exp_id}.")

    tokenizer = _get_tokenizer(run_id, exp_id)
    model = load_model_from_s3(exp_id, run_id, device)

    return model, tokenizer, device


def _get_tokenizer(run_id: str, exp_id: int):
    vocab = VocabularyWords()
    vocab.read_vocab(run_id, exp_id)
    return vocab.tokenizer


if __name__ == "__main__":
    model_prod, tokenizer_prod, device_prod = get_prod_model()
    get_sentiment("i like this film", model_prod, tokenizer_prod, device_prod)
