import csv
import numpy as np
from pso_engine_T2 import run_pso

N_RUNS = 30
SEED_BASE = 42

def run_multi_pso():
    print("Starting scientific multi-run experiment (Advanced PSO)...")

    all_histories = {
        "avg_fitness": [],
        "max_fitness": [],
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

        best_fit, history = run_pso(seed=seed)

        best_energy = history["best_energy"][-1]
        best_delay = history["best_delay"][-1]
        best_min_res = history["best_min_res"][-1]

        run_summaries.append([i+1, seed, best_fit, best_energy, best_delay, best_min_res])

        all_histories["avg_fitness"].append(history["avg_fitness"])
        all_histories["max_fitness"].append(history["max_fitness"])
        all_histories["best_energy"].append(history["best_energy"])
        all_histories["best_delay"].append(history["best_delay"])
        all_histories["best_min_res"].append(history["best_min_res"])

    print("\nCalculating statistical averages over all 30 runs...")

    avg_history = {k: np.mean(v, axis=0) for k, v in all_histories.items()}
    ITERATIONS = len(avg_history["max_fitness"])

    with open("pso_averaged_convergence_history.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Iteration", "MaxFitness", "AvgFitness", "TotalEnergy_J", "AvgDelay_Hops", "MinResidual_J"])
        for gen in range(ITERATIONS):
            writer.writerow([
                gen,
                avg_history["max_fitness"][gen],
                avg_history["avg_fitness"][gen],
                avg_history["best_energy"][gen],
                avg_history["best_delay"][gen],
                avg_history["best_min_res"][gen]
            ])

    with open("pso_multi_run_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Run", "Seed", "Best_Fitness", "Best_Energy_J", "Best_Delay_Hops", "Min_Residual_J"])
        for summary in run_summaries:
            writer.writerow(summary)

    print("\nMulti-run experiment finished successfully!")
    print("Important Files generated for Plotting:")
    print(" 1. pso_averaged_convergence_history.csv")
    print(" 2. pso_multi_run_summary.csv")

if __name__ == "__main__":
    run_multi_pso()