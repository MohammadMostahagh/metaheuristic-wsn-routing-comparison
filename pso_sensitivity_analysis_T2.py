import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from network import WSN
from particle_T2 import Particle, decode_position
from chromosome import has_cycle, repair_chromosome
from fitness import fitness


def calculate_swarm_diversity(swarm):
    """محاسبه فاصله همینگ (تنوع) بین درخت‌های مسیریابی ذرات"""
    total_distance = 0
    count = 0
    pop_len = len(swarm)
    for i in range(pop_len):
        for j in range(i + 1, pop_len):
            c1 = swarm[i].current_chromosome
            c2 = swarm[j].current_chromosome
            if len(c1) > 0 and len(c2) > 0:
                dist = sum(g1 != g2 for g1, g2 in zip(c1, c2))
                total_distance += dist
                count += 1
    return total_distance / count if count > 0 else 0


def run_custom_pso(swarm_size=80, c0=0.7, c1=1.5, c3=1.5, iterations=50):
    """اجرای یک باره PSO با پارامترهای دلخواه برای تحلیل حساسیت"""
    wsn = WSN(num_nodes=40, comm_radius=35)
    swarm = [Particle(wsn.num_nodes) for _ in range(swarm_size)]
    global_best_fitness = -1.0
    global_best_position = None

    diversity_history = []

    for it in range(iterations):
        for particle in swarm:
            chrom = decode_position(wsn, particle.position)
            if has_cycle(chrom):
                chrom = repair_chromosome(chrom, wsn)
            fit_val, _, _, _ = fitness(wsn, chrom)
            particle.current_chromosome = chrom
            particle.fitness = fit_val

            if fit_val > particle.pBest_fitness:
                particle.pBest_fitness = fit_val
                particle.pBest_position = particle.position.copy()
            if fit_val > global_best_fitness:
                global_best_fitness = fit_val
                global_best_position = particle.position.copy()

        diversity_history.append(calculate_swarm_diversity(swarm))

        for particle in swarm:
            r1, r3 = np.random.rand(wsn.num_nodes), np.random.rand(wsn.num_nodes)
            particle.velocity = (c0 * particle.velocity) + \
                                (c1 * r1 * (particle.pBest_position - particle.position)) + \
                                (c3 * r3 * (global_best_position - particle.position))
            particle.velocity = np.clip(particle.velocity, -0.5, 0.5)
            particle.position = np.abs((particle.position + particle.velocity) % 1.0)

    return global_best_fitness, diversity_history


def run_sensitivity_tests():
    print("Running PSO Sensitivity Analysis (This will take a minute)...")

    print(" -> 1/4 Testing Swarm Diversity...")
    _, div_history = run_custom_pso(swarm_size=80, c0=0.7, c1=1.5, c3=1.5, iterations=50)

    swarm_sizes = [10, 20, 40, 60, 80, 100, 120]
    swarm_fitness = []
    for s in swarm_sizes:
        print(f" -> 2/4 Testing Swarm Size: {s}...")
        fit, _ = run_custom_pso(swarm_size=s)
        swarm_fitness.append(fit)

    inertia_weights = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1]
    inertia_fitness = []
    for w in inertia_weights:
        print(f" -> 3/4 Testing Inertia Weight (C0): {w}...")
        fit, _ = run_custom_pso(c0=w)
        inertia_fitness.append(fit)

    learning_configs = [(2.5, 0.5), (2.0, 1.0), (1.5, 1.5), (1.0, 2.0), (0.5, 2.5)]
    labels_configs = ["High Personal", "Mid Personal", "Balanced", "Mid Social", "High Social"]
    learning_fitness = []
    for c1, c3 in learning_configs:
        print(f" -> 4/4 Testing C1={c1}, C3={c3}...")
        fit, _ = run_custom_pso(c1=c1, c3=c3)
        learning_fitness.append(fit)

    # ================= رسم 4 نمودار حساسیت و تنوع =================
    print("\nGenerating Plots...")
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("PSO Comprehensive Sensitivity Analysis & Diversity", fontsize=16, fontweight='bold')

    # نمودار تنوع (Diversity)
    axs[0, 0].plot(range(50), div_history, color='purple', linewidth=2)
    axs[0, 0].set_title('Swarm Diversity (Hamming Distance)', fontweight='bold')
    axs[0, 0].set_xlabel('Iteration')
    axs[0, 0].set_ylabel('Diversity Score')
    axs[0, 0].grid(True, linestyle=':')

    # نمودار سایز جمعیت
    axs[0, 1].plot(swarm_sizes, swarm_fitness, marker='o', color='c', linewidth=2, markersize=8)
    axs[0, 1].set_title('Effect of Swarm Size on Fitness', fontweight='bold')
    axs[0, 1].set_xlabel('Number of Particles')
    axs[0, 1].set_ylabel('Best Fitness Achieved')
    axs[0, 1].grid(True, linestyle=':')

    # نمودار ضریب اینرسی (C0)
    axs[1, 0].plot(inertia_weights, inertia_fitness, marker='s', color='m', linewidth=2, markersize=8)
    axs[1, 0].set_title('Effect of Inertia Weight (C0)', fontweight='bold')
    axs[1, 0].set_xlabel('Inertia Weight Value')
    axs[1, 0].set_ylabel('Best Fitness Achieved')
    axs[1, 0].grid(True, linestyle=':')

    # نمودار ضرایب یادگیری
    axs[1, 1].plot(labels_configs, learning_fitness, marker='^', color='orange', linewidth=2, markersize=8)
    axs[1, 1].set_title('Cognitive (C1) vs Social (C3) Focus', fontweight='bold')
    axs[1, 1].set_xlabel('Parameter Configuration')
    axs[1, 1].set_ylabel('Best Fitness Achieved')
    axs[1, 1].tick_params(axis='x', rotation=15)
    axs[1, 1].grid(True, linestyle=':')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("PSO_Sensitivity_Analysis.png", dpi=300)
    print("Success! Plots saved as 'PSO_Sensitivity_Analysis.png'")
    plt.show()


if __name__ == "__main__":
    run_sensitivity_tests()