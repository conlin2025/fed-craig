import os
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

from fed.datasets import get_cifar100_datasets
from fed.models import get_model
from fed.utils import set_seed

# =====================================================
# 🔧 HYPERPARAMETERS 
# =====================================================
DATA_DIR     = "./data"
BATCH_SIZE   = 128
NUM_EPOCHS   = 5        # small-ish; increase if you want more reliable scores
LR           = 0.01
SEED         = 123
OUT_PATH     = "./data/forgetting_scores_cifar100.npy"
MODEL_ARCH   = "resnet18"   # or whatever use in FL
DEVICE       = "cuda" if torch.cuda.is_available() else (
                 "mps" if torch.backends.mps.is_available() else "cpu"
              )
# =====================================================


class IndexedDataset(Dataset):
    """
    Wraps a base dataset and returns (x, y, idx),
    where idx is the global index into the base dataset.
    """
    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        return x, y, idx


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    criterion = nn.CrossEntropyLoss()

    for x, y, _ in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()


def compute_epoch_predictions(model, loader, device, n_samples):
    """
    Returns a boolean array 'correct_now' of shape (n_samples,)
    indicating whether each sample is currently classified correctly.
    """
    model.eval()
    correct_now = np.zeros(n_samples, dtype=bool)

    with torch.no_grad():
        for x, y, idx in loader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu()
            y = y.cpu()

            idx_np = idx.numpy()
            correct_batch = (preds == y).numpy()
            correct_now[idx_np] = correct_batch

    return correct_now


def main():
    print(">>> compute_forgetting_scores.py started")
    print("Using device:", DEVICE)

    set_seed(SEED)

    # 1. Load CIFAR-100 train split
    train_dataset, _ = get_cifar100_datasets(DATA_DIR)
    n = len(train_dataset)
    print(f"Train size: {n}")

    indexed_train = IndexedDataset(train_dataset)
    train_loader = DataLoader(
        indexed_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    score_loader = DataLoader(
        indexed_train,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # 2. Initialize model and optimizer
    model = get_model(num_classes=100, device=DEVICE, arch=MODEL_ARCH)
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)

    # 3. Initialize forgetting tracking arrays
    # prev_correct[i] : was sample i correct in previous evaluation?
    # forget_counts[i]: number of times i went from correct -> incorrect
    prev_correct = np.zeros(n, dtype=bool)
    forget_counts = np.zeros(n, dtype=np.int32)
    ever_seen = np.zeros(n, dtype=bool)   # not strictly necessary but nice to have

    print("Starting training + forgetting tracking...")
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        train_one_epoch(model, train_loader, optimizer, DEVICE)

        # Evaluate on training set to see current correctness
        correct_now = compute_epoch_predictions(model, score_loader, DEVICE, n)

        # Update forgetting counts
        for i in range(n):
            # Only count if sample was ever correct at some point
            if prev_correct[i] and not correct_now[i]:
                forget_counts[i] += 1

        # Update prev_correct and ever_seen
        ever_seen |= correct_now
        prev_correct = correct_now

        print(f"  Number of samples currently correct: {correct_now.sum()}")
        print(f"  Total forgetting events so far: {forget_counts.sum()}")

    # 4. Save forgetting scores
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    np.save(OUT_PATH, forget_counts)
    print(f"\nSaved forgetting scores to: {OUT_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()
