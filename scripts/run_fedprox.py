# scripts/run_fedprox.py
print(">>> run_fedprox.py started")

import numpy as np
import torch
import os
import csv

from fed.datasets import (
    get_cifar100_datasets,
    dirichlet_partition,
    get_client_loaders,
    get_test_loader,
    get_loader_from_indices,
)
from fed.models import get_model
from fed.algorithms import local_update_fedprox, aggregate_models
from fed.utils import evaluate, set_seed
from fed.selections import random_coreset, craig_like_coreset
from fed.datasets import get_loader_from_indices



# =====================================================
# 🔧 HYPERPARAMETERS — EDIT THESE VALUES DIRECTLY
# =====================================================

NUM_CLIENTS = 10            # number of simulated clients
ALPHA = 0.5                 # Dirichlet concentration; smaller = more non-IID
NUM_ROUNDS = 10             # communication rounds
LOCAL_EPOCHS = 1            # local epochs per client per round
LR = 0.01                   # local learning rate
MU = 0.001                  # FedProx proximal coefficient
FRAC_CLIENTS = 0.5          # fraction of clients sampled each round
BATCH_SIZE = 64             # batch size for local training
DATA_DIR = "./data"         # where CIFAR-100 is stored/downloaded
SEED = 42                   # RNG seed for reproducibility

USE_CORESET = True          # True: train on random coreset; False: full client data
CORESET_RATIO = 0.3         # fraction of each client's data to keep in coreset
CORESET_METHOD = "craig"   # "random" or "craig"
# =====================================================

RUN_NAME = "fedprox"


def main():
    print(">>> FedProx main() started")

    # Set seed
    set_seed(SEED)

    # Choose device: CUDA > MPS > CPU
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print("Using device:", device)

    # 1. Load CIFAR-100 datasets
    train_dataset, test_dataset = get_cifar100_datasets(DATA_DIR)
    print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")

    # 2. Create non-IID client partition using Dirichlet
    print(f"Creating Dirichlet partition with {NUM_CLIENTS} clients, alpha={ALPHA}")
    client_indices = dirichlet_partition(
        train_dataset,
        num_clients=NUM_CLIENTS,
        alpha=ALPHA,
        num_classes=100,
    )
    client_sizes = [len(v) for v in client_indices.values()]
    print("Client sizes:", client_sizes)

    # 3. Build DataLoaders for each client (full local data)
    client_loaders = get_client_loaders(
        train_dataset, client_indices, batch_size=BATCH_SIZE
    )
    test_loader = get_test_loader(test_dataset, batch_size=128)

    # 4. Initialize global model
    # If you already swapped get_model to use ResNet via arch="resnet18", you can pass that here:
    # global_model = get_model(num_classes=100, device=device, arch="resnet18")
    global_model = get_model(num_classes=100, device=device)
    print("Model initialized.")

    # 5. Federated training loop with FedProx local updates
    for rnd in range(1, NUM_ROUNDS + 1):
        print(f"\n--- FedProx Round {rnd} ---")
        global_model.train()

        # sample subset of clients
        m = max(1, int(FRAC_CLIENTS * NUM_CLIENTS))
        selected_clients = np.random.choice(range(NUM_CLIENTS), m, replace=False)
        print("Selected clients:", selected_clients)

        client_models = {}
        client_weights = {}

        # local update for each selected client
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

            # FedProx local update:
            # local objective = ERM + (MU/2) * ||w - w_global||^2
            local_model = local_update_fedprox(
                global_model,
                global_model,     # pass current global model for proximal term
                loader,
                device=device,
                epochs=LOCAL_EPOCHS,
                lr=LR,
                mu=MU,
            )

            client_models[cid] = local_model
            client_weights[cid] = len(client_indices[cid])

        # 6. Aggregate client models into new global model
        global_model = aggregate_models(global_model, client_models, client_weights)

        # 7. Evaluate on global test set
        test_loss, test_acc = evaluate(global_model, test_loader, device)
        print(f"Round {rnd}: test loss = {test_loss:.4f}, test acc = {test_acc:.4f}")


if __name__ == "__main__":
    main()
