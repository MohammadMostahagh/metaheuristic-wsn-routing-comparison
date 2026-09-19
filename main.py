import csv
import numpy as np
from ga import run_ga

N_RUNS = 30
SEED_BASE = 42

print("Starting scientific multi-run experiment (Advanced GA)...")

all_histories = {
    "avg_fitness": [],
    "max_fitness": [],
    "diversity": [],
    "best_energy": [],
    "best_delay": [],
    "best_min_res": []
}

run_summaries = []

for i in range(N_RUNS):
    seed = SEED_BASE + i
    print(f"\n{'='*40}")
    print(f" Run {i+1}/{N_RUNS} | Seed = {seed}")
    print(f"{'='*40}")

    best_fit, best_delay, best_energy, best_min_res, history = run_ga(seed=seed)

    run_summaries.append([i+1, seed, best_fit, best_energy, best_delay, best_min_res])

    all_histories["avg_fitness"].append(history["avg_fitness"])
    all_histories["max_fitness"].append(history["max_fitness"])
    all_histories["diversity"].append(history["diversity"])
    all_histories["best_energy"].append(history["best_energy"])
    all_histories["best_delay"].append(history["best_delay"])
    all_histories["best_min_res"].append(history["best_min_res"])


print("\n Calculating statistical averages over all runs...")

avg_history = {k: np.mean(v, axis=0) for k, v in all_histories.items()}

GENERATIONS = len(avg_history["max_fitness"])
with open("averaged_convergence_history.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Generation", "MaxFitness", "AvgFitness", "Diversity", "TotalEnergy_J", "AvgDelay_Hops", "MinResidual_J"])
    for gen in range(GENERATIONS):
        writer.writerow([
            gen,
            avg_history["max_fitness"][gen],
            avg_history["avg_fitness"][gen],
            avg_history["diversity"][gen],
            avg_history["best_energy"][gen],
            avg_history["best_delay"][gen],
            avg_history["best_min_res"][gen]
        ])

with open("multi_run_summary.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Run", "Seed", "Best_Fitness", "Best_Energy_J", "Best_Delay_Hops", "Min_Residual_J"])
    for summary in run_summaries:
        writer.writerow(summary)

print("\n Multi-run experiment finished successfully!")
print(" Important Files generated for Plotting:")
print("  1. averaged_convergence_history.csv")
print("  2. multi_run_summary.csv")