import random
import csv
import numpy as np

from network import WSN
from chromosome import generate_random_chromosome, has_cycle, repair_chromosome
from fitness import fitness

#POP_SIZE = 80
POP_SIZE = 100
#GENERATIONS = 50
GENERATIONS = 500
TOURNAMENT_K = 3
CROSSOVER_RATE = 0.9
MUTATION_RATE = 0.2


def tournament_selection(pop, fits):
    """انتخاب تورنمنت برای جلوگیری از همگرایی زودرس"""
    candidates = random.sample(list(zip(pop, fits)), TOURNAMENT_K)
    return max(candidates, key=lambda x: x[1])[0]


def uniform_crossover(p1, p2):
    """تقاطع یکنواخت برای ترکیب مسیرهای دو درخت مختلف"""
    child = []
    for g1, g2 in zip(p1, p2):
        child.append(g1 if random.random() < 0.5 else g2)
    return child


def mutate(chromosome, wsn):
    """
    جهش آگاه از توپولوژی شبکه
    فقط همسایگان مجاز که به سینک نزدیک‌ترند انتخاب می‌شوند
    """
    node_idx = random.randint(0, len(chromosome) - 1)
    node = wsn.nodes[node_idx]

    valid_next_hops = []
    for neighbor in node.neighbors:
        if neighbor == wsn.sink:
            valid_next_hops.append("sink")
        elif wsn.distance(neighbor, wsn.sink) < wsn.distance(node, wsn.sink):
            valid_next_hops.append(neighbor.id)

    if not valid_next_hops:
        valid_next_hops.append("sink")

    chromosome[node_idx] = random.choice(valid_next_hops)
    return chromosome


def population_diversity(population):
    """محاسبه تنوع ژنتیکی جمعیت (فاصله همینگ بین مسیرها)"""
    total_distance = 0
    count = 0
    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            c1 = population[i]
            c2 = population[j]
            dist = sum(g1 != g2 for g1, g2 in zip(c1, c2))
            total_distance += dist
            count += 1
    return total_distance / count if count > 0 else 0


def run_ga(seed=None):
    """ الگوریتم ژنتیک پیشرفته (SGA + Elitism + Repair)"""
    if seed is not None:
        random.seed(seed)

    wsn = WSN(num_nodes=40, comm_radius=35)
    ALGO_NAME = "Advanced_GA"

    population = [generate_random_chromosome(wsn) for _ in range(POP_SIZE)]

    history = {
        "avg_fitness": [],
        "max_fitness": [],
        "diversity": [],
        "best_delay": [],
        "best_energy": [],
        "best_min_res": []
    }

    for gen in range(GENERATIONS):
        # -------- ارزیابی جمعیت (Evaluation) --------
        fitness_results = [fitness(wsn, c) for c in population]
        fits = [f[0] for f in fitness_results]

        avg_fit = sum(fits) / len(fits)
        max_fit = max(fits)
        best_idx = fits.index(max_fit)

        best_individual = population[best_idx].copy()
        best_fit, best_delay, best_energy, best_min_res = fitness_results[best_idx]

        history["avg_fitness"].append(avg_fit)
        history["max_fitness"].append(max_fit)
        history["diversity"].append(population_diversity(population))
        history["best_delay"].append(best_delay)
        history["best_energy"].append(best_energy)
        history["best_min_res"].append(best_min_res)

        print(f"Gen {gen:02d} | Max Fit: {best_fit:.4f} | Energy: {best_energy:.4f} J | Delay: {best_delay:.2f}")

        new_population = []

        new_population.append(best_individual)

        while len(new_population) < POP_SIZE:
            p1 = tournament_selection(population, fits)
            p2 = tournament_selection(population, fits)

            if random.random() < CROSSOVER_RATE:
                child = uniform_crossover(p1, p2)
            else:
                child = p1.copy()

            if random.random() < MUTATION_RATE:
                child = mutate(child, wsn)

            if has_cycle(child):
                child = repair_chromosome(child, wsn)

            new_population.append(child)

        population = new_population

    # -------- ذخیره فایل‌های CSV برای نمودارها --------
    with open(f"{ALGO_NAME}_fitness_history.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Generation", "BestFitness", "AvgFitness", "Diversity"])
        for i in range(GENERATIONS):
            writer.writerow([i, history["max_fitness"][i], history["avg_fitness"][i], history["diversity"][i]])

    with open(f"{ALGO_NAME}_metrics_history.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Generation", "TotalEnergy_J", "AverageDelay_Hops", "MinResidual_J"])
        for i in range(GENERATIONS):
            writer.writerow([i, history["best_energy"][i], history["best_delay"][i], history["best_min_res"][i]])

    print(f"\n{ALGO_NAME} finished successfully! CSV logs saved.")
    return best_fit, best_delay, best_energy, best_min_res, history


if __name__ == "__main__":
    print("Running Advanced Genetic Algorithm (Pure GA)...")
    run_ga(seed=42)