# scripts/run_fedavg.py
print(">>> run_fedavg.py started")

import numpy as np
import torch
import os
import csv

from fed.datasets import (
    get_cifar100_datasets,
    dirichlet_partition,
    get_client_loaders,
    get_test_loader,
)
from fed.models import get_model
from fed.algorithms import local_update_fedavg, aggregate_models
from fed.utils import evaluate, set_seed
from fed.selections import random_coreset, craig_like_coreset
from fed.datasets import get_loader_from_indices



# =====================================================
# 🔧 HYPERPARAMETERS — CAN BE OVERRIDDEN BY ENV VARS
# =====================================================

NUM_CLIENTS = int(os.getenv("NUM_CLIENTS", "10"))
ALPHA = float(os.getenv("ALPHA", "0.5"))
NUM_ROUNDS = int(os.getenv("NUM_ROUNDS", "10"))
LOCAL_EPOCHS = int(os.getenv("LOCAL_EPOCHS", "1"))
LR = float(os.getenv("LR", "0.01"))
FRAC_CLIENTS = float(os.getenv("FRAC_CLIENTS", "0.5"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))
DATA_DIR = os.getenv("DATA_DIR", "./data")
SEED = int(os.getenv("SEED", "42"))

USE_CORESET = os.getenv("USE_CORESET", "0") == "1"   # "1" -> True, anything else -> False
CORESET_RATIO = float(os.getenv("CORESET_RATIO", "0.3"))
CORESET_METHOD = os.getenv("CORESET_METHOD", "random")  # "random" or "craig"

RUN_NAME = os.getenv("RUN_NAME", "fedavg_default")
# =====================================================


def main():
    algo_name = "fedavg"
    print(">>> FedAvg main() started")

    # Create results directory
    os.makedirs("results", exist_ok=True)

    # Define CSV path for this run
    csv_path = os.path.join("results", f"{RUN_NAME}.csv")

    # Overwrite file and write header
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "algo",          # fedavg or fedprox
            "coreset",       # full, random, craig, etc.
            "alpha",         # Dirichlet alpha
            "round",
            "test_loss",
            "test_acc",
        ])

    # Set seed
    set_seed(SEED)

    # detect device
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print("Using device:", device)

    # 1. Load CIFAR-100
    train_dataset, test_dataset = get_cifar100_datasets(DATA_DIR)
    print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")

    # 2. Dirichlet non-IID partition
    print(f"Creating Dirichlet partition with {NUM_CLIENTS} clients, alpha={ALPHA}")
    client_indices = dirichlet_partition(
        train_dataset,
        num_clients=NUM_CLIENTS,
        alpha=ALPHA,
        num_classes=100,
    )
    client_sizes = [len(v) for v in client_indices.values()]
    print("Client sizes:", client_sizes)

    # 3. Build client loaders
    client_loaders = get_client_loaders(train_dataset, client_indices, batch_size=BATCH_SIZE)
    test_loader = get_test_loader(test_dataset, batch_size=128)

    # 4. Initialize global model
    global_model = get_model(num_classes=100, device=device, arch="resnet18")
    print("Model initialized.")

    # 5. Federated training loop
    for rnd in range(1, NUM_ROUNDS + 1):
        print(f"\n--- FedAvg Round {rnd} ---")
        global_model.train()

        # sample a subset of clients
        m = max(1, int(FRAC_CLIENTS * NUM_CLIENTS))
        selected_clients = np.random.choice(range(NUM_CLIENTS), m, replace=False)
        print("Selected clients:", selected_clients)

        client_models = {}
        client_weights = {}

        # local update for each client
        for cid in selected_clients:
            print(f"  Training client {cid}")

            if USE_CORESET:
                full_indices = client_indices[cid]

                if CORESET_METHOD == "random":
                    coreset_indices = random_coreset(full_indices, ratio=CORESET_RATIO)
                elif CORESET_METHOD == "craig":
                    coreset_indices = craig_like_coreset(
                        global_model,
                        train_dataset,
                        full_indices,
                        ratio=CORESET_RATIO,
                        device=device,
                        batch_size=BATCH_SIZE,
                    )
                else:
                    raise ValueError(f"Unknown CORESET_METHOD: {CORESET_METHOD}")

                loader = get_loader_from_indices(
                    train_dataset,
                    coreset_indices,
                    batch_size=BATCH_SIZE,
                )
            else:
                loader = client_loaders[cid]


        # 6. Aggregate
        global_model = aggregate_models(global_model, client_models, client_weights)

        # 7. Evaluation
        test_loss, test_acc = evaluate(global_model, test_loader, device)
        print(f"Round {rnd}: test loss = {test_loss:.4f}, test acc = {test_acc:.4f}")

        # Determine coreset label
        if not USE_CORESET:
            coreset_label = "full"
        else:
            coreset_label = CORESET_METHOD  # e.g., "random" or "craig"

        # Append this round's result to CSV
        with open(csv_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                algo_name,
                coreset_label,
                ALPHA,
                rnd,
                test_loss,
                test_acc,
            ])


if __name__ == "__main__":
    main()
