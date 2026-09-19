import numpy as np
import random
import math
import csv
from network import WSN
from chromosome import generate_random_chromosome, has_cycle, repair_chromosome
from fitness import fitness

TEMP_STEPS = 500
#TEMP_STEPS = 50
MARKOV_CHAIN_LEN = 1000
#MARKOV_CHAIN_LEN = 80
T_INITIAL = 1.0
T_FINAL = 0.00001


# =================================================================================

def get_discrete_neighbor(wsn, current_chromosome):
    """
    تولید یک جواب همسایه در فضای کاملاً گسسته.
    مسیریابی یک گره تصادفی را تغییر می‌دهد.
    """
    neighbor = current_chromosome.copy()
    node_idx = random.randint(0, wsn.num_nodes - 1)
    node = wsn.nodes[node_idx]

    valid_next_hops = []
    for ngh in node.neighbors:
        if ngh == wsn.sink:
            valid_next_hops.append("sink")
        elif wsn.distance(ngh, wsn.sink) < wsn.distance(node, wsn.sink):
            valid_next_hops.append(ngh.id)

    if not valid_next_hops:
        valid_next_hops.append("sink")

    if len(valid_next_hops) > 1 and neighbor[node_idx] in valid_next_hops:
        valid_next_hops.remove(neighbor[node_idx])

    neighbor[node_idx] = random.choice(valid_next_hops)

    if has_cycle(neighbor):
        neighbor = repair_chromosome(neighbor, wsn)

    return neighbor


def run_sa(seed=42):
    """موتور اصلی الگوریتم شبیه‌سازی تبرید (SA)"""
    np.random.seed(seed)
    random.seed(seed)

    wsn = WSN(num_nodes=40, comm_radius=35)
    ALGO_NAME = "Advanced_SA"

    alpha = (T_FINAL / T_INITIAL) ** (1.0 / TEMP_STEPS)

    current_solution = generate_random_chromosome(wsn)
    if has_cycle(current_solution):
        current_solution = repair_chromosome(current_solution, wsn)

    current_fit, cur_delay, cur_eng, cur_min_res = fitness(wsn, current_solution)

    best_solution = current_solution.copy()
    best_fit = current_fit
    best_delay = cur_delay
    best_energy = cur_eng
    best_min_res = cur_min_res

    current_temp = T_INITIAL

    history = {
        "avg_fitness": [],
        "max_fitness": [],
        "best_delay": [],
        "best_energy": [],
        "best_min_res": []
    }

    print(f"Starting SA Engine (Temp Steps: {TEMP_STEPS}, Markov Chain: {MARKOV_CHAIN_LEN})...")

    for step in range(TEMP_STEPS):
        accepted_fitnesses = []

        for _ in range(MARKOV_CHAIN_LEN):
            neighbor = get_discrete_neighbor(wsn, current_solution)
            n_fit, n_del, n_eng, n_min_res = fitness(wsn, neighbor)

            delta = n_fit - current_fit

            if delta > 0:
                accept = True
            else:
                p_accept = math.exp(delta / current_temp)
                accept = (random.random() < p_accept)

            if accept:
                current_solution = neighbor.copy()
                current_fit = n_fit
                cur_delay = n_del
                cur_eng = n_eng
                cur_min_res = n_min_res

            accepted_fitnesses.append(current_fit)

            if current_fit > best_fit:
                best_solution = current_solution.copy()
                best_fit = current_fit
                best_delay = cur_delay
                best_energy = cur_eng
                best_min_res = cur_min_res

        avg_fit = sum(accepted_fitnesses) / len(accepted_fitnesses)

        history["avg_fitness"].append(avg_fit)
        history["max_fitness"].append(best_fit)
        history["best_delay"].append(best_delay)
        history["best_energy"].append(best_energy)
        history["best_min_res"].append(best_min_res)

        print(
            f"Temp Step {step:02d} | Temp: {current_temp:.4f} | Max Fit: {best_fit:.4f} | Avg Fit: {avg_fit:.4f} | Energy: {best_energy:.4f} J")

        current_temp *= alpha

    with open(f"{ALGO_NAME}_fitness_history.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Iteration", "BestFitness", "AvgFitness"])
        for i in range(TEMP_STEPS):
            writer.writerow([i, history["max_fitness"][i], history["avg_fitness"][i]])

    with open(f"{ALGO_NAME}_metrics_history.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Iteration", "TotalEnergy_J", "AverageDelay_Hops", "MinResidual_J"])
        for i in range(TEMP_STEPS):
            writer.writerow([i, history["best_energy"][i], history["best_delay"][i], history["best_min_res"][i]])

    print("\nSA Engine finished successfully!")
    return best_fit, history


if __name__ == "__main__":
    run_sa(seed=42)