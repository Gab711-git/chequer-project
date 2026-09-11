import torch
from torch.utils.data import Dataset


class PoisonedDataset(Dataset):
    def __init__(self, dataset):
        self.data = dataset.data.clone().float() / 255.0
        self.targets = dataset.targets.clone()
        self.poisoned_indices = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx].unsqueeze(0), self.targets[idx]


def label_flip(poisoned_dataset, source_label=1, target_label=7, poison_rate=0.30):
    source_indices = (poisoned_dataset.targets == source_label).nonzero(as_tuple=True)[0]
    n_poison = int(len(source_indices) * poison_rate)
    chosen = source_indices[torch.randperm(len(source_indices))[:n_poison]]

    poisoned_dataset.targets[chosen] = target_label
    poisoned_dataset.poisoned_indices.extend(chosen.tolist())

    print(f"Label flip: {n_poison} samples flipped ({source_label} -> {target_label})")
    return poisoned_dataset


def backdoor_inject(poisoned_dataset, source_label=3, target_label=8,
                    poison_rate=0.10, patch_size=4, row_start=16, col_start=12):
    source_indices = (poisoned_dataset.targets == source_label).nonzero(as_tuple=True)[0]
    n_poison = int(len(source_indices) * poison_rate)
    chosen = source_indices[torch.randperm(len(source_indices))[:n_poison]]

    for idx in chosen:
        poisoned_dataset.data[idx,
                              row_start:row_start + patch_size,
                              col_start:col_start + patch_size] = 1.0

    poisoned_dataset.targets[chosen] = target_label
    poisoned_dataset.poisoned_indices.extend(chosen.tolist())

    print(f"Backdoor inject: {n_poison} samples patched ({source_label} -> {target_label})")
    return poisoned_dataset
