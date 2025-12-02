import os
import csv
from collections import defaultdict

import matplotlib.pyplot as plt

RESULTS_DIR = "results"

# ================================
# FILTERS – EDIT THESE AS YOU LIKE
# ================================
# None means "no filter" (include all)
FILTER_ALGOS = ["fedavg", "fedprox"]        # ["fedavg"], ["fedprox"], or None
FILTER_CORESETS = ["full", "random", "craig"]  # subset or None
FILTER_ALPHAS = [0.1, 0.5]                  # [0.5], [0.1], or None
# ================================


def load_all_results():
    """
    Load all CSV files in results/ and group them by (algo, coreset, alpha).
    Expects each CSV to have columns:
      algo, coreset, alpha, round, test_loss, test_acc
    """
    config_to_rows = defaultdict(list)

    if not os.path.exists(RESULTS_DIR):
        return config_to_rows

    for fname in os.listdir(RESULTS_DIR):
        if not fname.endswith(".csv"):
            continue

        path = os.path.join(RESULTS_DIR, fname)
        with open(path, mode="r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                algo = row["algo"]
                coreset = row["coreset"]
                alpha = float(row["alpha"])
                rnd = int(row["round"])
                loss = float(row["test_loss"])
                acc = float(row["test_acc"])

                key = (algo, coreset, alpha)
                config_to_rows[key].append({
                    "round": rnd,
                    "test_loss": loss,
                    "test_acc": acc,
                })

    return config_to_rows


def aggregate_by_round(rows):
    """
    Given a list of rows (with 'round', 'test_loss', 'test_acc'),
    compute mean test_loss and test_acc per round across runs.
    """
    round_to_vals = defaultdict(list)
    for r in rows:
        rnd = r["round"]
        round_to_vals[rnd].append((r["test_loss"], r["test_acc"]))

    rounds = sorted(round_to_vals.keys())
    mean_losses = []
    mean_accs = []

    for rnd in rounds:
        vals = round_to_vals[rnd]
        losses = [x[0] for x in vals]
        accs = [x[1] for x in vals]
        mean_losses.append(sum(losses) / len(losses))
        mean_accs.append(sum(accs) / len(accs))

    return rounds, mean_losses, mean_accs


def passes_filters(algo, coreset, alpha):
    if FILTER_ALGOS is not None and algo not in FILTER_ALGOS:
        return False
    if FILTER_CORESETS is not None and coreset not in FILTER_CORESETS:
        return False
    if FILTER_ALPHAS is not None and alpha not in FILTER_ALPHAS:
        return False
    return True


def plot_avg_accuracy(config_to_rows,
                      title="Test Accuracy vs Rounds (averaged)",
                      save_path=None):
    plt.figure(figsize=(8, 5))

    any_plotted = False

    for (algo, coreset, alpha), rows in config_to_rows.items():
        if not rows:
            continue

        if not passes_filters(algo, coreset, alpha):
            continue

        rounds, _, mean_accs = aggregate_by_round(rows)
        label = f"{algo} | {coreset} | alpha={alpha}"
        plt.plot(rounds, mean_accs, marker="o", label=label)
        any_plotted = True

    if not any_plotted:
        print("No configs matched the filters. Adjust FILTER_* in this script.")
        return

    plt.xlabel("Round")
    plt.ylabel("Test Accuracy")
    plt.title(title)
    plt.grid(True)
    plt.legend()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print("Saved plot to:", save_path)
    else:
        plt.show()


def main():
    config_to_rows = load_all_results()
    if not config_to_rows:
        print("No results found in 'results/'")
        return

    out_path = os.path.join(RESULTS_DIR, "avg_comparison.png")
    plot_avg_accuracy(
        config_to_rows,
        title="CIFAR-100 FedAvg/FedProx – mean over runs",
        save_path=out_path,
    )


if __name__ == "__main__":
    main()
