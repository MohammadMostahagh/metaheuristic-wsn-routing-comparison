import numpy as np
import random
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
from network import WSN
from chromosome import generate_random_chromosome, has_cycle, repair_chromosome
from fitness import fitness

SWARM_SIZE = 30
ITERATIONS = 50


def evaluate_pso_params(c0, c1, c2, seed=42):
    np.random.seed(seed)
    random.seed(seed)
    wsn = WSN(num_nodes=40, comm_radius=35)

    swarm_pos = []
    swarm_vel = []
    pbest_pos = []
    pbest_fit = []

    gbest_pos = []
    gbest_fit = -float('inf')

    for _ in range(SWARM_SIZE):
        chrom = generate_random_chromosome(wsn)
        if has_cycle(chrom): chrom = repair_chromosome(chrom, wsn)
        fit, _, _, _ = fitness(wsn, chrom)

        swarm_pos.append(chrom)
        swarm_vel.append(np.random.uniform(-1, 1, wsn.num_nodes))
        pbest_pos.append(chrom.copy())
        pbest_fit.append(fit)
        if fit > gbest_fit:
            gbest_fit = fit
            gbest_pos = chrom.copy()

    for _ in range(ITERATIONS):
        for i in range(SWARM_SIZE):
            r1 = np.random.rand(wsn.num_nodes)
            r2 = np.random.rand(wsn.num_nodes)

            for j in range(wsn.num_nodes):
                pos_val = wsn.num_nodes if swarm_pos[i][j] == "sink" else swarm_pos[i][j]
                pbest_val = wsn.num_nodes if pbest_pos[i][j] == "sink" else pbest_pos[i][j]
                gbest_val = wsn.num_nodes if gbest_pos[j] == "sink" else gbest_pos[j]

                cog = c1 * r1[j] * (pbest_val - pos_val)
                soc = c2 * r2[j] * (gbest_val - pos_val)
                swarm_vel[i][j] = c0 * swarm_vel[i][j] + cog + soc

            new_pos = []
            for j in range(wsn.num_nodes):
                sig = 1.0 / (1.0 + np.exp(-swarm_vel[i][j]))
                if np.random.rand() < sig:
                    valid = [ngh.id if ngh != wsn.sink else "sink" for ngh in wsn.nodes[j].neighbors
                             if ngh == wsn.sink or wsn.distance(ngh, wsn.sink) < wsn.distance(wsn.nodes[j], wsn.sink)]
                    if not valid: valid = ["sink"]
                    new_pos.append(random.choice(valid))
                else:
                    new_pos.append(swarm_pos[i][j])

            if has_cycle(new_pos): new_pos = repair_chromosome(new_pos, wsn)
            fit, _, _, _ = fitness(wsn, new_pos)
            swarm_pos[i] = new_pos.copy()

            if fit > pbest_fit[i]:
                pbest_fit[i] = fit
                pbest_pos[i] = new_pos.copy()
                if fit > gbest_fit: gbest_fit = fit

    return gbest_fit


def tune_pso():
    print("Starting Grid Search for PSO Parameters...")
    C0_values = [0.4, 0.7, 0.9]
    C1_values = [0.5, 1.5, 2.5]
    C2_values = [0.5, 1.5, 2.5]

    results = []
    total_configs = len(C0_values) * len(C1_values) * len(C2_values)
    count = 0

    for c0 in C0_values:
        for c1 in C1_values:
            for c2 in C2_values:
                count += 1
                fit = evaluate_pso_params(c0, c1, c2)
                results.append({'C0': c0, 'C1': c1, 'C2': c2, 'Fitness': fit})
                print(f"Tested {count}/{total_configs}: C0={c0}, C1={c1}, C2={c2} -> Fit: {fit:.4f}")

    df = pd.DataFrame(results)
    best_config = df.loc[df['Fitness'].idxmax()]
    best_c0 = best_config['C0']
    print(
        f"\nBEST PSO CONFIGURATION: C0={best_config['C0']}, C1={best_config['C1']}, C2={best_config['C2']} -> Fitness: {best_config['Fitness']:.4f}")

    df_filtered = df[df['C0'] == best_c0]
    pivot_table = df_filtered.pivot(index='C1', columns='C2', values='Fitness')

    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(pivot_table.values, cmap="YlGnBu", aspect="auto")
    fig.colorbar(cax)

    for i in range(len(pivot_table.index)):
        for j in range(len(pivot_table.columns)):
            ax.text(j, i, f"{pivot_table.values[i, j]:.4f}", ha="center", va="center",
                    color="black" if pivot_table.values[i, j] > pivot_table.values.mean() else "white")

    ax.set_xticks(np.arange(len(pivot_table.columns)))
    ax.set_yticks(np.arange(len(pivot_table.index)))
    ax.set_xticklabels(pivot_table.columns)
    ax.set_yticklabels(pivot_table.index)

    plt.title(f"PSO Parameter Tuning (Fixed C0 = {best_c0})", fontweight='bold')
    plt.xlabel("C2 (Social - Global Best)")
    plt.ylabel("C1 (Cognitive - Local Best)")
    plt.tight_layout()
    plt.savefig("PSO_Tuning_Heatmap.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    tune_pso()