# scripts/run_batch.py

"""
Run + clear results + auto-plot:
CLEAR_RESULTS=1 PLOT_AFTER=1 python -m scripts.run_batch
"""

import os
import glob

# =====================================================
# 🔧 GLOBAL EXPERIMENT DEFAULTS (for real runs)
# =====================================================
NUM_CLIENTS   = 10
NUM_ROUNDS    = 20
LOCAL_EPOCHS  = 1
LR            = 0.01
MU            = 0.001          # FedProx μ
FRAC_CLIENTS  = 0.5
BATCH_SIZE    = 64
DATA_DIR      = "./data"

ALPHAS        = [0.5]     # non-IID severity
SEEDS         = [42, 43, 44]   # seeds per config
CORESET_RATIO = 0.3            # 30% coresets
# =====================================================

# (algo, coreset_type, use_coreset)
CORESET_CONFIGS = [
    ("fedavg",  "full",       False),
    ("fedavg",  "forgetting", True),
    ("fedavg",  "sieve",      True),
    ("fedprox", "full",       False),
    ("fedprox", "forgetting", True),
    ("fedprox", "sieve",      True),
]


def maybe_clear_results():
    """Optionally clear old CSVs and plot if CLEAR_RESULTS=1."""
    clear_flag = os.getenv("CLEAR_RESULTS", "0")
    if clear_flag != "1":
        print("[INFO] CLEAR_RESULTS != 1, keeping existing results/")
        return

    os.makedirs("results", exist_ok=True)
    for f in glob.glob("results/*.csv"):
        os.remove(f)
    png_path = os.path.join("results", "avg_comparison.png")
    if os.path.exists(png_path):
        os.remove(png_path)

    print("[INFO] Cleared old CSVs and avg_comparison.png in results/")


def maybe_plot_after():
    """Optionally call scripts.plot_avg_results if PLOT_AFTER=1."""
    plot_flag = os.getenv("PLOT_AFTER", "0")
    if plot_flag != "1":
        print("[INFO] PLOT_AFTER != 1, skipping automatic plotting")
        return

    print("[INFO] Calling scripts.plot_avg_results to generate plot...")
    try:
        from scripts.plot_avg_results import main as plot_main
        plot_main()
    except Exception as e:
        print("[WARN] Failed to run plot_avg_results:", e)


def run_experiment(cfg):
    """
    cfg = (algo, coreset_type, alpha, seed, use_coreset)
    """
    algo, coreset_type, alpha, seed, use_coreset = cfg

    run_name = f"{algo}_{coreset_type}_a{alpha}_seed{seed}".replace(".", "_")

    print("\n" + "=" * 60)
    print(f"Running experiment: {run_name}")
    print(f"  Algo   : {algo}")
    print(f"  Alpha  : {alpha}")
    print(f"  Coreset: {coreset_type} (use_coreset={use_coreset}, ratio={CORESET_RATIO})")
    print(f"  Seed   : {seed}")
    print("=" * 60)

    # Set env vars for this run
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
    os.environ["CORESET_METHOD"] = coreset_type   # "full" ignored when USE_CORESET=0
    os.environ["CORESET_RATIO"]  = str(CORESET_RATIO)

    os.environ["MU"]            = str(MU)         # used by FedProx, harmless for FedAvg

    # Call appropriate script
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
    maybe_clear_results()

    all_cfgs = []
    for alpha in ALPHAS:
        for seed in SEEDS:
            for algo, coreset_type, use_coreset in CORESET_CONFIGS:
<<<<<<< HEAD
                # use global CORESET_RATIO when the config requests a coreset
                coreset_ratio = CORESET_RATIO if use_coreset else 0.0
                all_cfgs.append((algo, coreset_type, alpha, seed, use_coreset, coreset_ratio))
=======
                all_cfgs.append(
                    (algo, coreset_type, alpha, seed, use_coreset)
                )
>>>>>>> acf4461 (update plot_automated)

    print(f"Total runs to execute: {len(all_cfgs)}")
    for cfg in all_cfgs:
        run_experiment(cfg)

    maybe_plot_after()


if __name__ == "__main__":
    main()
