import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import csv

def plot_scientific_results():
    generations = []
    max_fitness = []
    avg_fitness = []
    diversity = []
    total_energy = []
    avg_delay = []
    min_residual = []

    try:
        with open("averaged_convergence_history.csv", "r") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                generations.append(int(row[0]))
                max_fitness.append(float(row[1]))
                avg_fitness.append(float(row[2]))
                diversity.append(float(row[3]))
                total_energy.append(float(row[4]))
                avg_delay.append(float(row[5]))
                min_residual.append(float(row[6]))
    except FileNotFoundError:
        print("Error: 'averaged_convergence_history.csv' not found. Please run main.py first.")
        return

    plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

    # ==========================================
    # ==========================================
    plt.figure(figsize=(8, 6))
    plt.plot(generations, max_fitness, label="Max Fitness (Best Individual)", color='#1f77b4', linewidth=2.5)
    plt.plot(generations, avg_fitness, label="Average Fitness", color='#ff7f0e', linewidth=2.5, linestyle='--')
    plt.xlabel("Generation", fontweight='bold')
    plt.ylabel("Fitness Value", fontweight='bold')
    plt.title("Convergence of Fitness Function over Generations", fontweight='bold', fontsize=14)
    plt.legend(loc="lower right", shadow=True)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig("plot_1_fitness_convergence.png", dpi=300)
    plt.show()

    # ==========================================
    # ==========================================
    plt.figure(figsize=(8, 6))
    plt.plot(generations, diversity, label="Population Diversity (Hamming Distance)", color='#9467bd', linewidth=2.5)
    plt.xlabel("Generation", fontweight='bold')
    plt.ylabel("Diversity Score", fontweight='bold')
    plt.title("Genetic Diversity of Population During Evolution", fontweight='bold', fontsize=14)
    plt.legend(loc="upper right", shadow=True)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig("plot_2_population_diversity.png", dpi=300)
    plt.show()

    # ==========================================
    # ==========================================
    plt.figure(figsize=(8, 6))
    plt.plot(generations, total_energy, label="Total Energy Consumed (Joules)", color='#2ca02c', linewidth=2.5)
    plt.xlabel("Generation", fontweight='bold')
    plt.ylabel("Energy (Joules)", fontweight='bold')
    plt.title("Optimization of Network Energy Consumption", fontweight='bold', fontsize=14)
    plt.legend(loc="upper right", shadow=True)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig("plot_3_energy_optimization.png", dpi=300)
    plt.show()

    # ==========================================
    # ==========================================
    plt.figure(figsize=(8, 6))
    plt.plot(generations, avg_delay, label="Average End-to-End Delay (Hops)", color='#d62728', linewidth=2.5)
    plt.xlabel("Generation", fontweight='bold')
    plt.ylabel("Delay (Number of Hops)", fontweight='bold')
    plt.title("Optimization of Routing Delay", fontweight='bold', fontsize=14)
    plt.legend(loc="upper right", shadow=True)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig("plot_4_delay_optimization.png", dpi=300)
    plt.show()

    # ==========================================
    # ==========================================
    plt.figure(figsize=(8, 6))
    plt.plot(generations, min_residual, label="Minimum Residual Energy (Joules)", color='#17becf', linewidth=2.5)
    plt.xlabel("Generation", fontweight='bold')
    plt.ylabel("Residual Energy (Joules)", fontweight='bold')
    plt.title("Load Balancing: Protection of Bottleneck Nodes", fontweight='bold', fontsize=14)
    plt.legend(loc="lower right", shadow=True)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig("plot_5_residual_energy.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    print("Reading averaged_convergence_history.csv...")
    print("Generating High-Resolution Scientific Plots...")
    plot_scientific_results()
    print("All 5 plots saved successfully as PNGs!")