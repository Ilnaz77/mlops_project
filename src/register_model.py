import os

import torch
import mlflow
from mlflow import MlflowClient
from mlflow.entities import ViewType
from torch.utils.data import DataLoader

from src.model import load_model_from_s3
from src.dataloader import Collate, QueryDataset, VocabularyWords


def update_production_model():
    mlflow.set_tracking_uri(f"http://{os.environ['TRACKING_SERVER_HOST']}:5000")
    client = MlflowClient()

    # Retrieve the top_n model runs and log the models
    experiment = client.get_experiment_by_name(os.environ["MLFLOW_EXPERIMENT_NAME"])

    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=5,
        order_by=["metrics.best_val_loss ASC"],
    )

    best_model = {"mean_test_loss": 1e34, "artifacts_run_id": None}
    for run in runs:
        if run.info.status != "FINISHED":
            continue
        loss = test(experiment.experiment_id, run.info.run_id)
        if loss < best_model["mean_test_loss"]:
            best_model["mean_test_loss"] = loss
            best_model["artifacts_run_id"] = run.info.run_id

    # Register the best model
    version = mlflow.register_model(
        model_uri=f"runs:/{best_model['artifacts_run_id']}/model",
        name=os.environ["MODEL_NAME"],
        tags={
            "exp_id": experiment.experiment_id,
            "exp_name": os.environ["MLFLOW_EXPERIMENT_NAME"],
        },
    )

    client.transition_model_version_stage(
        name=os.environ["MODEL_NAME"],
        version=version.version,
        stage="Production",
        archive_existing_versions=True,
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
        shuffle=False,
    )

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
    update_production_model()
