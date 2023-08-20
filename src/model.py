import os

import torch
import mlflow
from torch import nn
from mlflow import MlflowClient
from torch.nn.utils.rnn import pack_padded_sequence


class RNNModel(nn.Module):
    def __init__(
        self,
        output_size: int,
        vocab_size: int,
        embed_size: int,
        hidden_size: int,
        pad_idx: int,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.bidirectional = True
        self.hidden_size = hidden_size if not self.bidirectional else hidden_size // 2

        self.embeddings = nn.Embedding(vocab_size, embed_size, padding_idx=self.pad_idx)

        self.rnn = nn.LSTM(
            embed_size,
            self.hidden_size,
            batch_first=True,
            bidirectional=self.bidirectional,
        )

        self.fc = nn.Linear((1 + self.bidirectional) * self.hidden_size, output_size)

    def forward(self, x: torch.Tensor, seq_lengths: torch.Tensor):
        sequences = self.embeddings(x)  # ((batch_size X seq_len X embedding_dim))
        packed_input = pack_padded_sequence(sequences, seq_lengths, batch_first=True)

        _, (ht, ct) = self.rnn(packed_input)

        hidden = torch.cat([ht[0, :, :], ht[1, :, :]], dim=1) if self.bidirectional else ht.squeeze(0)

        output = self.fc(hidden)

        return output

    def inference(self, x: torch.Tensor):
        sequences = self.embeddings(x)
        _, (ht, ct) = self.rnn(sequences)
        hidden = torch.cat([ht[0, :, :], ht[1, :, :]], dim=1) if self.bidirectional else ht.squeeze(0)
        output = self.fc(hidden)
        return output


def load_model_from_s3(exp_id: int, run_id: str, device: torch.device):
    state_dict = f"s3://{os.environ['BUCKET_NAME']}/{exp_id}/{run_id}/artifacts/{os.environ['MODEL_ARTIFACT_PATH']}/"
    loaded_state_dict = mlflow.pytorch.load_state_dict(state_dict, map_location=device)

    client = MlflowClient(tracking_uri=f"http://{os.environ['TRACKING_SERVER_HOST']}:5000")
    params = client.get_run(run_id).data.to_dictionary()

    model = RNNModel(
        vocab_size=int(params["params"]["vocab_word_size"]),
        output_size=int(params["params"]["output_size"]),
        embed_size=int(params["params"]["embed_size"]),
        hidden_size=int(params["params"]["hidden_size"]),
        pad_idx=int(params["params"]["pad_idx"]),
    )
    model.load_state_dict(loaded_state_dict)

    return model.eval()
