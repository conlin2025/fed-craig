# scripts/run_test_batch.py

"""
Run + clear results + auto-plot:
CLEAR_RESULTS=1 PLOT_AFTER=1 python -m scripts.run_test_batch
"""

import os
import glob

"""
Tiny sanity-check batch.
Runs:
    fedavg  - full
    fedavg  - forgetting
    fedavg  - sieve
    fedprox - full
    fedprox - forgetting
    fedprox - sieve

Tiny settings → fast debugging.

Environment flags:
    CLEAR_RESULTS=1  -> remove old results/*.csv and results/avg_comparison.png
    PLOT_AFTER=1     -> automatically call scripts.plot_avg_results at the end
"""

# =====================================================
# 🔧 SUPER SMALL DEBUG-FRIENDLY SETTINGS
# =====================================================
NUM_CLIENTS   = 2        # very small → each client has tiny data
NUM_ROUNDS    = 1        # one communication round
LOCAL_EPOCHS  = 1        # one local step
LR            = 0.01
MU            = 0.001
FRAC_CLIENTS  = 1.0      # use all clients (only 2)
BATCH_SIZE    = 16
DATA_DIR      = "./data"
ALPHAS        = [0.5]    # 1 non-IID level
SEEDS         = [0]      # 1 seed
CORESET_RATIO = 0.5      # each coreset half of the tiny dataset
# =====================================================

# Methods to test
TEST_CONFIGS = [
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
    # Import and call main() from plot_avg_results
    try:
        from scripts.plot_avg_results import main as plot_main
        plot_main()
    except Exception as e:
        print("[WARN] Failed to run plot_avg_results:", e)


def run_experiment(cfg):
    algo, coreset_type, use_coreset, alpha, seed = cfg

    run_name = f"TEST_{algo}_{coreset_type}_a{alpha}_seed{seed}".replace(".", "_")

    print("\n" + "=" * 60)
    print(f"[TEST] Running: {run_name}")
    print(f"  Algo     : {algo}")
    print(f"  Coreset  : {coreset_type} (use={use_coreset}, ratio={CORESET_RATIO})")
    print(f"  Alpha    : {alpha}")
    print(f"  Seed     : {seed}")
    print("=" * 60)

    # Set env vars for run_fedavg / run_fedprox
    os.environ["RUN_NAME"]       = run_name
    os.environ["NUM_CLIENTS"]    = str(NUM_CLIENTS)
    os.environ["ALPHA"]          = str(alpha)
    os.environ["NUM_ROUNDS"]     = str(NUM_ROUNDS)
    os.environ["LOCAL_EPOCHS"]   = str(LOCAL_EPOCHS)
    os.environ["LR"]             = str(LR)
    os.environ["FRAC_CLIENTS"]   = str(FRAC_CLIENTS)
    os.environ["BATCH_SIZE"]     = str(BATCH_SIZE)
    os.environ["DATA_DIR"]       = DATA_DIR
    os.environ["SEED"]           = str(seed)

    os.environ["USE_CORESET"]    = "1" if use_coreset else "0"
    os.environ["CORESET_METHOD"] = coreset_type
    os.environ["CORESET_RATIO"]  = str(CORESET_RATIO)

    os.environ["MU"]             = str(MU)

    # which script?
    if algo == "fedavg":
        cmd = "python -m scripts.run_fedavg"
    elif algo == "fedprox":
        cmd = "python -m scripts.run_fedprox"
    else:
        raise ValueError(f"Unknown algo: {algo}")

    print(f"[EXEC] {cmd}")
    code = os.system(cmd)
    if code != 0:
        print(f"[ERROR] Run {run_name} exited with code {code}")


def main():
    maybe_clear_results()

    all_cfgs = []
    for alpha in ALPHAS:
        for seed in SEEDS:
            for algo, coreset_type, use_coreset in TEST_CONFIGS:
                cfg = (algo, coreset_type, use_coreset, alpha, seed)
                all_cfgs.append(cfg)

    print(f"[TEST] Total runs: {len(all_cfgs)}")
    for cfg in all_cfgs:
        run_experiment(cfg)

    maybe_plot_after()


if __name__ == "__main__":
    main()
