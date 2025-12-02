import os

# =====================================================
# 🔧 HYPERPARAMETERS NOTATIONS (GLOBAL DEFAULTS HERE)
# =====================================================
# NUM_CLIENTS   : number of simulated clients
# ALPHA         : Dirichlet concentration; smaller ⇒ more non-IID
# NUM_ROUNDS    : total communication rounds
# LOCAL_EPOCHS  : local epochs per client per round
# LR            : local learning rate (client SGD)
# MU            : FedProx proximal coefficient μ
# FRAC_CLIENTS  : fraction of clients sampled each round
# BATCH_SIZE    : local training batch size
# DATA_DIR      : where CIFAR-100 is stored/downloaded
#
# NOTE:
# - These are the "real experiment" settings.
# - run_fedavg.py / run_fedprox.py have SMALL defaults for debugging,
#   but run_batch.py overrides them via environment variables.
# =====================================================

NUM_CLIENTS   = 10
NUM_ROUNDS    = 20
LOCAL_EPOCHS  = 1
LR            = 0.01
MU            = 0.001      # FedProx μ
FRAC_CLIENTS  = 0.5
BATCH_SIZE    = 64
DATA_DIR      = "./data"

# Experiment grid:
ALPHAS = [0.1, 0.5]        # non-IID severity levels to test
SEEDS  = [42, 43, 44]      # seeds per config (for averaging)

# (algo, coreset_type, use_coreset, coreset_ratio)
# coreset_type will be written into "coreset" column in CSV:
#   "full"   : USE_CORESET=False (coreset_ratio ignored)
#   "random" : random subset
#   "craig"  : CRAIG-lite feature-based subset
CORESET_CONFIGS = [
    ("fedavg",  "full",   False, 0.0),
    ("fedavg",  "random", True,  0.3),
    ("fedavg",  "craig",  True,  0.3),
    ("fedprox", "full",   False, 0.0),
    ("fedprox", "random", True,  0.3),
    ("fedprox", "craig",  True,  0.3),
]


def run_experiment(cfg):
    """
    cfg = (algo, coreset_type, alpha, seed, use_coreset, coreset_ratio)
    """
    algo, coreset_type, alpha, seed, use_coreset, coreset_ratio = cfg

    # Construct a unique run name, e.g.:
    #   fedavg_random_a0_5_seed42
    run_name = f"{algo}_{coreset_type}_a{alpha}_seed{seed}".replace(".", "_")

    print("\n" + "=" * 60)
    print(f"Running experiment: {run_name}")
    print(f"  Algo   : {algo}")
    print(f"  Alpha  : {alpha}")
    print(f"  Coreset: {coreset_type} "
          f"(use_coreset={use_coreset}, ratio={coreset_ratio})")
    print(f"  Seed   : {seed}")
    print("=" * 60)

    # Set env vars for this run (used by run_fedavg / run_fedprox)
    os.environ["RUN_NAME"]      = run_name
    os.environ["NUM_CLIENTS"]   = str(NUM_CLIENTS)
    os.environ["ALPHA"]         = str(alpha)
    os.environ["NUM_ROUNDS"]    = str(NUM_ROUNDS)
    os.environ["LOCAL_EPOCHS"]  = str(LOCAL_EPOCHS)
    os.environ["LR"]            = str(LR)
    os.environ["FRAC_CLIENTS"]  = str(FRAC_CLIENTS)
    os.environ["BATCH_SIZE"]    = str(BATCH_SIZE)
    os.environ["DATA_DIR"]      = DATA_DIR
    os.environ["SEED"]          = str(seed)

    os.environ["USE_CORESET"]   = "1" if use_coreset else "0"
    os.environ["CORESET_METHOD"] = coreset_type  # "full" is ignored if USE_CORESET=0
    os.environ["CORESET_RATIO"]  = str(coreset_ratio)

    os.environ["MU"]            = str(MU)        # used by FedProx, harmless for FedAvg

    # Choose which script to run
    if algo == "fedavg":
        cmd = "python -m scripts.run_fedavg"
    elif algo == "fedprox":
        cmd = "python -m scripts.run_fedprox"
    else:
        raise ValueError(f"Unknown algo: {algo}")

    print(f"Executing: {cmd}")
    code = os.system(cmd)
    if code != 0:
        print(f"Run {run_name} exited with code {code}")


def main():
    all_cfgs = []
    for alpha in ALPHAS:
        for seed in SEEDS:
            for algo, coreset_type, use_coreset, coreset_ratio in CORESET_CONFIGS:
                all_cfgs.append(
                    (algo, coreset_type, alpha, seed, use_coreset, coreset_ratio)
                )

    print(f"Total runs to execute: {len(all_cfgs)}")
    for cfg in all_cfgs:
        run_experiment(cfg)


if __name__ == "__main__":
    main()
