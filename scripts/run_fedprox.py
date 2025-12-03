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
from fed.selections import (
    random_coreset,
    craig_like_coreset,
    forgetting_coreset,
    sieve_streaming_coreset,
)


# =====================================================
# 🔧 HYPERPARAMETER NOTATION
# -----------------------------------------------------
# NUM_CLIENTS    : number of simulated clients
# ALPHA          : Dirichlet concentration; smaller ⇒ more non-IID
# NUM_ROUNDS     : total communication rounds
# LOCAL_EPOCHS   : local epochs per client per round
# LR             : local learning rate (client SGD)
# MU             : FedProx proximal coefficient μ
# FRAC_CLIENTS   : fraction of clients sampled each round
# BATCH_SIZE     : local training batch size
# DATA_DIR       : where CIFAR-100 is stored/downloaded
# SEED           : RNG seed for reproducibility
# USE_CORESET    : if True, train on a subset (coreset) of each client's data
# CORESET_RATIO  : fraction of each client’s data kept in the coreset
# CORESET_METHOD : "random" or "craig" (CRAIG-lite) or "forgetting" or "sieve"
# RUN_NAME       : name used for the CSV log file in results/
# =====================================================

NUM_CLIENTS   = int(os.getenv("NUM_CLIENTS", "4"))
ALPHA         = float(os.getenv("ALPHA", "0.5"))
NUM_ROUNDS    = int(os.getenv("NUM_ROUNDS", "3"))
LOCAL_EPOCHS  = int(os.getenv("LOCAL_EPOCHS", "1"))
LR            = float(os.getenv("LR", "0.01"))
MU            = float(os.getenv("MU", "0.001"))          # FedProx μ
FRAC_CLIENTS  = float(os.getenv("FRAC_CLIENTS", "0.5"))
BATCH_SIZE    = int(os.getenv("BATCH_SIZE", "32"))
DATA_DIR      = os.getenv("DATA_DIR", "./data")
SEED          = int(os.getenv("SEED", "42"))

USE_CORESET     = os.getenv("USE_CORESET", "0") == "1"
CORESET_RATIO   = float(os.getenv("CORESET_RATIO", "0.5"))
CORESET_METHOD  = os.getenv("CORESET_METHOD", "random")
FORGETTING_PATH = os.getenv("FORGETTING_PATH", "./data/forgetting_scores_cifar100.npy")
RUN_NAME        = os.getenv("RUN_NAME", "fedprox_debug")

# Optional speedup flags for sieve selector (env vars)
# REDUCE_DIM: int (e.g. 256) -> apply random projection to this many dimensions
# USE_APPROX_NN: "1" to force Annoy-based approximate NN, "0" to force exact NN,
#               unset -> leave selector default (selector will enable Annoy by default)
REDUCE_DIM_RAW = os.getenv("REDUCE_DIM", "")
REDUCE_DIM = int(REDUCE_DIM_RAW) if REDUCE_DIM_RAW != "" else None
USE_APPROX_NN_RAW = os.getenv("USE_APPROX_NN", "")
if USE_APPROX_NN_RAW == "":
    USE_APPROX_NN = None
else:
    USE_APPROX_NN = USE_APPROX_NN_RAW == "1"

# =====================================================


def main():
    algo_name = "fedprox"
    print(">>> FedProx main() started")

    # Create results directory and ensure CSV header exists.
    os.makedirs("results", exist_ok=True)
    csv_path = os.path.join("results", f"{RUN_NAME}.csv")

    header = [
        "algo",
        "coreset",
        "alpha",
        "round",
        "test_loss",
        "test_acc",
    ]

    # Ensure header exists. Use append mode but write header only when file
    # does not exist or is empty to avoid accidental truncation.
    if (not os.path.exists(csv_path)) or (os.path.getsize(csv_path) == 0):
        with open(csv_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

    # Set seed
    set_seed(SEED)

    # Device
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

    # 4. Initialize global model (use same ResNet-18 as FedAvg for parity)
    global_model = get_model(num_classes=100, device=device, arch="resnet18")
    print("Model initialized.")

    # Print Prox and sieve-related config for easy debugging in Colab
    print(f"MU={MU}, REDUCE_DIM={REDUCE_DIM}, USE_APPROX_NN={USE_APPROX_NN}")

    # 4.5 Load forgetting scores if needed
    forget_scores = None
    if USE_CORESET and CORESET_METHOD == "forgetting":
        print(f"Loading forgetting scores from: {FORGETTING_PATH}")
        forget_scores = np.load(FORGETTING_PATH)
        if forget_scores.shape[0] != len(train_dataset):
            raise ValueError(
                f"Forgetting scores length {forget_scores.shape[0]} "
                f"does not match train dataset size {len(train_dataset)}"
            )
        print("Forgetting scores loaded.")

    # 5. Federated training loop with FedProx local updates
    # Simple per-client feature cache to avoid recomputing features each round
    feature_cache = {}
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
                    coreset_indices = random_coreset(
                        full_indices, ratio=CORESET_RATIO
                    )

                elif CORESET_METHOD == "craig":
                    coreset_indices = craig_like_coreset(
                        global_model,
                        train_dataset,
                        full_indices,
                        ratio=CORESET_RATIO,
                        device=device,
                        batch_size=BATCH_SIZE,
                    )

                elif CORESET_METHOD == "forgetting":
                    if forget_scores is None:
                        raise RuntimeError(
                            "forget_scores is None but CORESET_METHOD='forgetting'. "
                            "Did you run compute_forgetting_scores and set FORGETTING_PATH?"
                        )
                    coreset_indices = forgetting_coreset(
                        full_indices,
                        forgetting_scores=forget_scores,
                        ratio=CORESET_RATIO,
                        pick="high",
                    )

                elif CORESET_METHOD == "sieve":
                    pre_feats = feature_cache.get(cid)
                    coreset_indices = sieve_streaming_coreset(
                        global_model,
                        train_dataset,
                        full_indices,
                        ratio=CORESET_RATIO,
                        device=device,
                        batch_size=BATCH_SIZE,
                        precomputed_features=pre_feats,
                        reduce_dim=REDUCE_DIM,
                        use_approx_nn=USE_APPROX_NN,
                    )
                    if pre_feats is None:
                        from fed.selections import _extract_feature_matrix
                        feature_cache[cid] = _extract_feature_matrix(
                            global_model,
                            train_dataset,
                            full_indices,
                            device,
                            batch_size=BATCH_SIZE,
                            reduce_dim=None,
                            random_state=None,
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
            if USE_CORESET:
               client_weights[cid] = len(coreset_indices)
            else:
               client_weights[cid] = len(client_indices[cid])

        # 6. Aggregate client models into new global model
        global_model = aggregate_models(global_model, client_models, client_weights)

        # 7. Evaluate on global test set
        test_loss, test_acc = evaluate(global_model, test_loader, device)
        print(f"Round {rnd}: test loss = {test_loss:.4f}, test acc = {test_acc:.4f}")

        if not USE_CORESET:
            coreset_label = "full"
        else:
            coreset_label = CORESET_METHOD

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
