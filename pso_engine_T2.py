import numpy as np
import random
import csv
from network import WSN
from chromosome import has_cycle, repair_chromosome
from fitness import fitness

from particle_T2 import Particle, decode_position

SWARM_SIZE = 100
#SWARM_SIZE = 80
ITERATIONS = 500
#ITERATIONS = 50
C0 = 0.4
C1 = 2.5
C3 = 0.5
V_MAX = 0.5


# =================================================================================

def run_pso(seed=42):
    """ الگوریتم PSO برای شبکه حسگر"""
    np.random.seed(seed)
    random.seed(seed)

    wsn = WSN(num_nodes=40, comm_radius=35)
    ALGO_NAME = "Advanced_PSO"

    swarm = [Particle(wsn.num_nodes) for _ in range(SWARM_SIZE)]

    global_best_position = None
    global_best_fitness = -1.0
    global_best_chromosome = []

    g_best_delay = 0.0
    g_best_energy = 0.0
    g_best_min_res = 0.0

    history = {
        "avg_fitness": [],
        "max_fitness": [],
        "best_delay": [],
        "best_energy": [],
        "best_min_res": []
    }

    print(f"Starting PSO Engine (Swarm: {SWARM_SIZE}, Iterations: {ITERATIONS})...")

    for it in range(ITERATIONS):
        iteration_fitnesses = []

        # -------- فاز ارزیابی (Evaluation) --------
        for particle in swarm:
            chrom = decode_position(wsn, particle.position)

            if has_cycle(chrom):
                chrom = repair_chromosome(chrom, wsn)

            fit_val, delay, energy, min_res = fitness(wsn, chrom)
            particle.current_chromosome = chrom
            particle.fitness = fit_val
            iteration_fitnesses.append(fit_val)

            if fit_val > particle.pBest_fitness:
                particle.pBest_fitness = fit_val
                particle.pBest_position = particle.position.copy()

            if fit_val > global_best_fitness:
                global_best_fitness = fit_val
                global_best_position = particle.position.copy()
                global_best_chromosome = chrom.copy()
                g_best_delay = delay
                g_best_energy = energy
                g_best_min_res = min_res

        # -------- فاز حرکت (Movement & Update) --------
        for particle in swarm:
            r1 = np.random.rand(wsn.num_nodes)
            r3 = np.random.rand(wsn.num_nodes)

            particle.velocity = (C0 * particle.velocity) + \
                                (C1 * r1 * (particle.pBest_position - particle.position)) + \
                                (C3 * r3 * (global_best_position - particle.position))

            particle.velocity = np.clip(particle.velocity, -V_MAX, V_MAX)

            particle.position = particle.position + particle.velocity

            particle.position = np.abs(particle.position % 1.0)

        avg_fit = sum(iteration_fitnesses) / len(iteration_fitnesses)

        history["avg_fitness"].append(avg_fit)
        history["max_fitness"].append(global_best_fitness)
        history["best_delay"].append(g_best_delay)
        history["best_energy"].append(g_best_energy)
        history["best_min_res"].append(g_best_min_res)

        print(
            f"Iter {it:02d} | Max Fit: {global_best_fitness:.4f} | Avg Fit: {avg_fit:.4f} | Energy: {g_best_energy:.4f} J | Delay: {g_best_delay:.2f}")

    # -------- ذخیره فایل های CSV --------
    with open(f"{ALGO_NAME}_fitness_history.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Iteration", "BestFitness", "AvgFitness"])
        for i in range(ITERATIONS):
            writer.writerow([i, history["max_fitness"][i], history["avg_fitness"][i]])

    with open(f"{ALGO_NAME}_metrics_history.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Iteration", "TotalEnergy_J", "AverageDelay_Hops", "MinResidual_J"])
        for i in range(ITERATIONS):
            writer.writerow([i, history["best_energy"][i], history["best_delay"][i], history["best_min_res"][i]])

    print("\nPSO finished successfully! CSV logs saved.")
    return global_best_fitness, history


if __name__ == "__main__":
    run_pso(seed=42)