import os

import mlflow
import torch
from torch import nn
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

        self.rnn = nn.LSTM(embed_size, self.hidden_size, batch_first=True, bidirectional=self.bidirectional)

        self.fc = nn.Linear((1 + self.bidirectional) * self.hidden_size, output_size)

    def forward(self, x: torch.Tensor, seq_lengths: torch.Tensor):
        sequences = self.embeddings(x)  # ((batch_size X seq_len X embedding_dim))
        packed_input = pack_padded_sequence(sequences, seq_lengths, batch_first=True)

        output, (ht, ct) = self.rnn(packed_input)

        hidden = torch.cat([ht[0, :, :], ht[1, :, :]], dim=1) if self.bidirectional else ht.squeeze(0)

        output = self.fc(hidden)

        return output


def load_model_from_s3(exp_id: int, run_id: str, device: torch.device):
    logged_model = f"s3://{os.environ['BUCKET_NAME']}/{exp_id}/{run_id}/artifacts/{os.environ['MODEL_ARTIFACT_PATH']}/"
    loaded_model = mlflow.pytorch.load_model(logged_model, map_location=device).eval()
    return loaded_model
