import os

""""
# =====================================================
# 🔧 HYPERPARAMETERS NOTATIONS
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
"""

# List of experiment configs
# You can edit this list to add/remove experiments.
# Each dict is ONE run (one seed/config).
experiments = [
    # --- FedAvg, full data, 3 seeds ---
    {
        "algo": "fedavg",
        "run_name": "fedavg_full_seed1",
        "alpha": 0.5,
        "use_coreset": False,
        "coreset_method": "random",   # ignored if use_coreset=False
        "coreset_ratio": 0.0,
        "seed": 42,
    },
    {
        "algo": "fedavg",
        "run_name": "fedavg_full_seed2",
        "alpha": 0.5,
        "use_coreset": False,
        "coreset_method": "random",
        "coreset_ratio": 0.0,
        "seed": 43,
    },
    {
        "algo": "fedavg",
        "run_name": "fedavg_full_seed3",
        "alpha": 0.5,
        "use_coreset": False,
        "coreset_method": "random",
        "coreset_ratio": 0.0,
        "seed": 44,
    },

    # --- FedAvg, random coreset 30%, 3 seeds ---
    {
        "algo": "fedavg",
        "run_name": "fedavg_random_0_3_seed1",
        "alpha": 0.5,
        "use_coreset": True,
        "coreset_method": "random",
        "coreset_ratio": 0.3,
        "seed": 52,
    },
    {
        "algo": "fedavg",
        "run_name": "fedavg_random_0_3_seed2",
        "alpha": 0.5,
        "use_coreset": True,
        "coreset_method": "random",
        "coreset_ratio": 0.3,
        "seed": 53,
    },
    {
        "algo": "fedavg",
        "run_name": "fedavg_random_0_3_seed3",
        "alpha": 0.5,
        "use_coreset": True,
        "coreset_method": "random",
        "coreset_ratio": 0.3,
        "seed": 54,
    },

    # --- FedProx, CRAIG-lite coreset 30%, 3 seeds (example) ---
    {
        "algo": "fedprox",
        "run_name": "fedprox_craig_0_3_seed1",
        "alpha": 0.5,
        "use_coreset": True,
        "coreset_method": "craig",
        "coreset_ratio": 0.3,
        "seed": 62,
    },
    {
        "algo": "fedprox",
        "run_name": "fedprox_craig_0_3_seed2",
        "alpha": 0.5,
        "use_coreset": True,
        "coreset_method": "craig",
        "coreset_ratio": 0.3,
        "seed": 63,
    },
    {
        "algo": "fedprox",
        "run_name": "fedprox_craig_0_3_seed3",
        "alpha": 0.5,
        "use_coreset": True,
        "coreset_method": "craig",
        "coreset_ratio": 0.3,
        "seed": 64,
    },
]

# Global defaults (you can tune once here)
NUM_CLIENTS = 10
NUM_ROUNDS = 10
LOCAL_EPOCHS = 1
LR = 0.01
MU = 0.001         # used only for FedProx
FRAC_CLIENTS = 0.5
BATCH_SIZE = 64
DATA_DIR = "./data"


def run_experiment(cfg):
    print("\n" + "=" * 60)
    print(f"Running experiment: {cfg['run_name']}")
    print(f"  Algo: {cfg['algo']}")
    print(f"  Alpha: {cfg['alpha']}")
    print(f"  Coreset: {'full' if not cfg['use_coreset'] else cfg['coreset_method']} "
          f"({cfg['coreset_ratio']})")
    print(f"  Seed: {cfg['seed']}")
    print("=" * 60)

    # Set env vars for this run
    os.environ["RUN_NAME"] = cfg["run_name"]
    os.environ["NUM_CLIENTS"] = str(NUM_CLIENTS)
    os.environ["ALPHA"] = str(cfg["alpha"])
    os.environ["NUM_ROUNDS"] = str(NUM_ROUNDS)
    os.environ["LOCAL_EPOCHS"] = str(LOCAL_EPOCHS)
    os.environ["LR"] = str(LR)
    os.environ["FRAC_CLIENTS"] = str(FRAC_CLIENTS)
    os.environ["BATCH_SIZE"] = str(BATCH_SIZE)
    os.environ["DATA_DIR"] = DATA_DIR
    os.environ["SEED"] = str(cfg["seed"])

    os.environ["USE_CORESET"] = "1" if cfg["use_coreset"] else "0"
    os.environ["CORESET_METHOD"] = cfg["coreset_method"]
    os.environ["CORESET_RATIO"] = str(cfg["coreset_ratio"])

    # FedProx-specific μ
    os.environ["MU"] = str(MU)

    # Call the appropriate training script
    if cfg["algo"] == "fedavg":
        cmd = "python -m scripts.run_fedavg"
    elif cfg["algo"] == "fedprox":
        cmd = "python -m scripts.run_fedprox"
    else:
        raise ValueError(f"Unknown algo: {cfg['algo']}")

    print(f"Executing: {cmd}")
    code = os.system(cmd)
    if code != 0:
        print(f"Run {cfg['run_name']} exited with code {code}")


def main():
    for cfg in experiments:
        run_experiment(cfg)


if __name__ == "__main__":
    main()
