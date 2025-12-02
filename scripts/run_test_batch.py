import os

# =====================================================
# 🔧 TINY DEBUG BATCH – for quick sanity checks
# =====================================================
# These are intentionally SMALL so you can verify:
# - env var wiring
# - CSV logging
# - plotting
# without waiting forever.
# =====================================================

NUM_CLIENTS   = 2
NUM_ROUNDS    = 2
LOCAL_EPOCHS  = 1
LR            = 0.01
MU            = 0.001
FRAC_CLIENTS  = 0.5
BATCH_SIZE    = 16
DATA_DIR      = "./data"

ALPHAS = [0.5]          # just one alpha for quick test
SEEDS  = [0]            # single seed

# A very small set of configs:
CORESET_CONFIGS = [
    ("fedavg",  "full",   False, 0.0),   # FedAvg, full data
    ("fedavg",  "random", True,  0.5),   # FedAvg, random coreset (50%)
    ("fedprox", "random", True,  0.5),   # FedProx, random coreset (50%)
]


def run_experiment(cfg):
    algo, coreset_type, alpha, seed, use_coreset, coreset_ratio = cfg

    run_name = f"TEST_{algo}_{coreset_type}_a{alpha}_seed{seed}".replace(".", "_")

    print("\n" + "=" * 60)
    print(f"[TEST] Running experiment: {run_name}")
    print(f"  Algo   : {algo}")
    print(f"  Alpha  : {alpha}")
    print(f"  Coreset: {coreset_type} "
          f"(use_coreset={use_coreset}, ratio={coreset_ratio})")
    print(f"  Seed   : {seed}")
    print("=" * 60)

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
    os.environ["CORESET_METHOD"] = coreset_type
    os.environ["CORESET_RATIO"]  = str(coreset_ratio)

    os.environ["MU"]            = str(MU)

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

    print(f"[TEST] Total runs to execute: {len(all_cfgs)}")
    for cfg in all_cfgs:
        run_experiment(cfg)


if __name__ == "__main__":
    main()
