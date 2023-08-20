from __future__ import annotations

import os
import json
import shutil
from typing import Any, Dict, List

import torch
import pandas as pd
from torch import LongTensor
from tokenizers import Tokenizer
from torch.utils.data import Dataset
from tokenizers.models import WordLevel
from torch.nn.utils.rnn import pad_sequence
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from src.utils import read_parquet_s3, read_pickle_from_s3


class Collate:
    def __init__(self, pad_idx: int = 0):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        words = [torch.tensor(item["text"]) for item in batch]
        labels = [item["labels"] for item in batch]

        # idxes to sort by sent len
        seq_lengths, perm_idx = LongTensor(list(map(len, words))).sort(0, descending=True)

        # sort by len sentence in desc
        words = [words[i] for i in perm_idx]
        labels = [labels[i] for i in perm_idx]

        # pad words: [batch_size, max_sentence_len_in_batch]
        padd_words = pad_sequence(words, padding_value=self.pad_idx, batch_first=True)

        return {
            "words": padd_words,
            "labels": torch.LongTensor(labels),
            "seq_lengths": seq_lengths.cpu().numpy(),
        }


class QueryDataset(Dataset):
    def __init__(self, file_path: str, vocab_words: VocabularyWords):
        self.data: pd.DataFrame = read_parquet_s3(file_path)
        self.vocab_words = vocab_words

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        line = self.data.iloc[index]
        label: int = line["sentiment"]
        text: str = line["text"]
        numericalized_word: List[int] = self.vocab_words.numericalize(text)

        return {"text": numericalized_word, "labels": label}


class VocabularyWords:
    def __init__(self, n_freq: int = 5, batch_size: int = 5000):
        self.stoi = dict()
        self.n_freq = n_freq
        self.batch_size = batch_size
        self.special_tokens = ["<PAD>", "<UNK>"]

        self.tokenizer = Tokenizer(WordLevel(unk_token="<UNK>"))
        self.tokenizer.pre_tokenizer = Whitespace()

    def __len__(self) -> int:
        return self.tokenizer.get_vocab_size()

    def save_vocab(self, mlflow):
        if os.path.isdir("./temp/"):
            shutil.rmtree("./temp/")
        os.makedirs("./temp/")
        self.tokenizer.save('./temp/vocab.json')
        mlflow.log_artifact('./temp/vocab.json', os.environ["VOCAB_ARTIFACT_PATH"])
        shutil.rmtree("./temp/")

    def read_vocab(self, run_id: str, exp_id: int):
        path = f"s3://{os.environ['BUCKET_NAME']}/{exp_id}/{run_id}/artifacts/{os.environ['VOCAB_ARTIFACT_PATH']}/vocab.json"
        if os.path.isdir("./temp/"):
            shutil.rmtree("./temp/")
        os.makedirs("./temp/")
        with open('./temp/tokenizer.json', 'w') as f:
            json.dump(read_pickle_from_s3(path), f)
        self.tokenizer = Tokenizer.from_file('./temp/tokenizer.json')
        self.stoi = self.tokenizer.get_vocab()
        shutil.rmtree("./temp/")

    def _get_training_corpus(self, sents: List[str]) -> List[str]:
        len_sents = len(sents)
        for i in range(0, len_sents, self.batch_size):
            yield sents[i : i + self.batch_size]

    def build_vocabulary(self, sentences_list: List[str]):
        trainer = WordLevelTrainer(
            special_tokens=self.special_tokens,
            min_frequency=self.n_freq,
        )
        self.tokenizer.train_from_iterator(self._get_training_corpus(sentences_list), trainer=trainer)
        self.stoi = self.tokenizer.get_vocab()

    def numericalize(self, text: str) -> List[int]:
        encoding = self.tokenizer.encode(text)
        return encoding.ids
