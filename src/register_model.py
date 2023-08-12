import os

import mlflow
import numpy as np
import torch
from mlflow import MlflowClient
from mlflow.entities import ViewType
from torch.utils.data import DataLoader

from dataloader import Collate, QueryDataset, VocabularyWords
from model import load_model_from_s3


def run_register_model():
    mlflow.set_tracking_uri(f"http://{os.environ['TRACKING_SERVER_HOST']}:5000")
    mlflow.set_experiment(os.environ["MLFLOW_TEST_EXPERIMENT_NAME"])

    client = MlflowClient()

    # Retrieve the top_n model runs and log the models
    experiment = client.get_experiment_by_name(os.environ["MLFLOW_TRAIN_EXPERIMENT_NAME"])
    train_experiment_id = experiment.experiment_id

    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=5,
        order_by=["metrics.best_val_loss ASC"]
    )

    for run in runs:
        if run.info.status != "FINISHED":
            continue
        with mlflow.start_run():
            loss = test(experiment.experiment_id, run.info.run_id)
            mlflow.log_metric("mean_test_loss", loss)
            mlflow.log_param("artifacts_run_id", run.info.run_id)

    # Select the model with the lowest test RMSE
    experiment = client.get_experiment_by_name(os.environ["MLFLOW_TEST_EXPERIMENT_NAME"])
    best_run = client.search_runs(experiment_ids=experiment.experiment_id,
                                  run_view_type=ViewType.ACTIVE_ONLY,
                                  order_by=["metrics.mean_test_loss ASC"])[0]

    # Register the best model
    version = mlflow.register_model(model_uri=f"runs:/{best_run.info.run_id}/model",
                                    name=f"sentiment_analys_model",
                                    tags={"exp_id": train_experiment_id,
                                          "run_id": best_run.data.params['artifacts_run_id']})

    client.transition_model_version_stage(name="sentiment_analys_model", version=version.version, stage="Staging")


def update_production_model():
    client = MlflowClient(tracking_uri=f"http://{os.environ['TRACKING_SERVER_HOST']}:5000")
    model_name = "sentiment_analys_model"

    best_version = -1
    best_loss = np.inf
    for mv in client.search_model_versions(f"name='{model_name}'"):
        exp_id = mv.tags["exp_id"]
        run_id = mv.tags["run_id"]
        version = mv.version
        loss = test(exp_id, run_id)
        if loss < best_loss:
            best_loss = loss
            best_version = version

    client.transition_model_version_stage(
        name=model_name,
        version=best_version,
        stage="Production",
        archive_existing_versions=True
    )


def test(exp_id: int, run_id: str) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab_words = VocabularyWords()
    test_dataset = QueryDataset(file_path=os.environ["TEST_PATH"], vocab_words=vocab_words)
    vocab_words.read_vocab(run_id, exp_id)

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=40,
        collate_fn=Collate(pad_idx=vocab_words.stoi["<PAD>"]),
        shuffle=False)

    model = load_model_from_s3(exp_id, run_id, device)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=vocab_words.stoi["<PAD>"], reduction="sum")

    model.eval()
    mean_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            words, labels, seq_lengths = (
                batch["words"].to(device),
                batch["labels"].to(device),
                batch["seq_lengths"],
            )
            predictions = model(words, seq_lengths)
            loss = criterion(predictions, labels)
            mean_loss += loss.cpu().item()

    return mean_loss / len(test_dataset)


if __name__ == "__main__":
    run_register_model()
    update_production_model()
