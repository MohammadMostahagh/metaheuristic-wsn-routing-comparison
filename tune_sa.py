import numpy as np
import random
import math
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
from network import WSN
from chromosome import generate_random_chromosome, has_cycle, repair_chromosome
from fitness import fitness

TEMP_STEPS = 100
MARKOV_CHAIN_LEN = 50


def evaluate_sa_params(alpha_rate, t_final, seed=42):
    np.random.seed(seed)
    random.seed(seed)
    wsn = WSN(num_nodes=40, comm_radius=35)

    current_sol = generate_random_chromosome(wsn)
    if has_cycle(current_sol): current_sol = repair_chromosome(current_sol, wsn)
    current_fit, _, _, _ = fitness(wsn, current_sol)

    best_fit = current_fit
    t_initial = 1.0
    current_temp = t_initial

    for _ in range(TEMP_STEPS):
        for _ in range(MARKOV_CHAIN_LEN):
            node_idx = random.randint(0, wsn.num_nodes - 1)
            neighbor = current_sol.copy()
            node = wsn.nodes[node_idx]

            valid = [ngh.id if ngh != wsn.sink else "sink" for ngh in node.neighbors
                     if ngh == wsn.sink or wsn.distance(ngh, wsn.sink) < wsn.distance(node, wsn.sink)]
            if not valid: valid = ["sink"]

            old_p = neighbor[node_idx]
            if len(valid) > 1 and old_p in valid: valid.remove(old_p)
            neighbor[node_idx] = random.choice(valid)

            if has_cycle(neighbor): neighbor = repair_chromosome(neighbor, wsn)

            n_fit, _, _, _ = fitness(wsn, neighbor)
            delta = n_fit - current_fit

            if current_temp < 1e-10:
                prob = 1.0 if delta > 0 else 0.0
            else:
                try:
                    prob = math.exp(delta / current_temp)
                except OverflowError:
                    prob = 0.0

            if delta > 0 or random.random() < prob:
                current_sol, current_fit = neighbor.copy(), n_fit
                if current_fit > best_fit: best_fit = current_fit

        if alpha_rate is not None:
            current_temp *= alpha_rate
        else:
            alpha = (t_final / t_initial) ** (1.0 / TEMP_STEPS)
            current_temp *= alpha

    return best_fit


def tune_sa():
    print("Starting Grid Search for SA Parameters...")
    T_final_values = [0.1, 0.01, 0.001, 0.00001 , 0.000001]
    Alpha_values = [0.85, 0.90, 0.95, 0.99]

    results = []

    for t_f in T_final_values:
        fit = evaluate_sa_params(alpha_rate=None, t_final=t_f)
        results.append({'Strategy': f'Target T_final={t_f}', 'Fitness': fit})
        print(f"Tested Target T_final={t_f} -> Fit: {fit:.4f}")

    for a in Alpha_values:
        fit = evaluate_sa_params(alpha_rate=a, t_final=1e-10)
        results.append({'Strategy': f'Fixed Alpha={a}', 'Fitness': fit})
        print(f"Tested Fixed Alpha={a} -> Fit: {fit:.4f}")

    df = pd.DataFrame(results)
    best_config = df.loc[df['Fitness'].idxmax()]
    print(f"\nBEST SA CONFIGURATION: {best_config['Strategy']} -> Fitness: {best_config['Fitness']:.4f}")

    plt.figure(figsize=(10, 6))
    plt.bar(df['Strategy'], df['Fitness'], color='skyblue', edgecolor='black')
    plt.xticks(rotation=45, ha='right')

    min_fit = df['Fitness'].min()
    max_fit = df['Fitness'].max()
    margin = (max_fit - min_fit) * 0.1 if max_fit != min_fit else 0.01
    plt.ylim(min_fit - margin, max_fit + margin)

    plt.title("SA Parameter Tuning (Final Fitness comparison)", fontweight='bold')
    plt.ylabel("Best Fitness")
    plt.tight_layout()
    plt.savefig("SA_Tuning_Barplot.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    tune_sa()