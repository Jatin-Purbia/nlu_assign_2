import os
from typing import Tuple, List

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
PAD_TOKEN = "<PAD>"


def load_names(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip().title() for line in f if line.strip()]


def build_vocab(names: List[str]) -> Tuple[dict, dict, List[str]]:
    unique_chars = sorted(set(ch for name in names for ch in name))
    vocab    = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN] + unique_chars
    char2idx = {ch: idx for idx, ch in enumerate(vocab)}
    idx2char = {idx: ch for ch, idx in char2idx.items()}
    return char2idx, idx2char, vocab


class NameDataset(Dataset):
    def __init__(self, names_file: str):
        self.names    = load_names(names_file)
        self.char2idx, self.idx2char, self.vocab = build_vocab(self.names)
        self.vocab_size = len(self.vocab)
        self._encoded   = [self._encode(name) for name in self.names]

    def _encode(self, name: str) -> torch.Tensor:
        return torch.tensor([self.char2idx[ch] for ch in name], dtype=torch.long)

    def encode_name(self, name: str) -> torch.Tensor:
        return self._encode(name)

    def decode_tensor(self, tensor: torch.Tensor) -> str:
        skip = {self.char2idx[PAD_TOKEN], self.char2idx[SOS_TOKEN], self.char2idx[EOS_TOKEN]}
        return "".join(self.idx2char[idx.item()] for idx in tensor if idx.item() not in skip)

    def __len__(self) -> int:
        return len(self._encoded)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded    = self._encoded[idx]
        sos        = torch.tensor([self.char2idx[SOS_TOKEN]], dtype=torch.long)
        eos        = torch.tensor([self.char2idx[EOS_TOKEN]], dtype=torch.long)
        input_seq  = torch.cat([sos, encoded])
        target_seq = torch.cat([encoded, eos])
        return input_seq, target_seq


def collate_fn(batch):
    inputs, targets = zip(*batch)
    lengths         = torch.tensor([len(seq) for seq in inputs], dtype=torch.long)
    inputs_padded   = pad_sequence(inputs,  batch_first=True, padding_value=0)
    targets_padded  = pad_sequence(targets, batch_first=True, padding_value=0)
    return inputs_padded, targets_padded, lengths


def get_dataloader(names_file, batch_size=32, shuffle=True):
    dataset = NameDataset(names_file)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                         collate_fn=collate_fn, drop_last=False)
    return loader, dataset


if __name__ == "__main__":
    names_path = os.path.join(os.path.dirname(__file__), "..", "TrainingNames.txt")
    loader, ds = get_dataloader(names_path, batch_size=4)
    print(f"Vocab size  : {ds.vocab_size}")
    print(f"Dataset size: {len(ds)} names")
    inputs, targets, lengths = next(iter(loader))
    print(f"Batch shapes -- inputs: {inputs.shape}, targets: {targets.shape}")
    print(f"Example decoded: {ds.decode_tensor(inputs[0])}")
