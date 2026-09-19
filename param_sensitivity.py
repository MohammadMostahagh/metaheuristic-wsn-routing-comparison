import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import ga


def test_mutation_sensitivity():
    print("\n" + "=" * 40)
    print(" Running Parameter Sensitivity: MUTATION RATE")
    print("=" * 40)

    mutation_rates = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25 , 0.3 , 0.35 , 0.4 , 0.45 , 0.5]
    best_fits = []

    ga.POP_SIZE = 80

    for m in mutation_rates:
        print(f"\n--> Testing Mutation Rate = {m}")
        ga.MUTATION_RATE = m
        fit, delay, energy, min_res, _ = ga.run_ga(seed=42)
        best_fits.append(fit)

    plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})
    plt.figure(figsize=(8, 6))
    plt.plot(mutation_rates, best_fits, marker="s", markersize=8, color="#e377c2", linewidth=2.5)
    plt.xlabel("Mutation Rate (Probability)", fontweight='bold')
    plt.ylabel("Best Fitness Achieved", fontweight='bold')
    plt.title("Sensitivity Analysis: Effect of Mutation Rate on Fitness", fontweight='bold', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.7)

    for i, txt in enumerate(best_fits):
        plt.annotate(f"{txt:.4f}", (mutation_rates[i], best_fits[i]), textcoords="offset points", xytext=(0, 10),
                     ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig("plot_5_mutation_sensitivity.png", dpi=300)
    print(" Mutation sensitivity plot saved as 'plot_5_mutation_sensitivity.png'")
    plt.show()


def test_population_sensitivity():
    print("\n" + "=" * 40)
    print(" Running Parameter Sensitivity: POPULATION SIZE")
    print("=" * 40)

    pop_sizes = [10, 20, 40, 60, 80, 100 , 110 , 120 , 130 , 140, 150]
    best_fits = []

    ga.MUTATION_RATE = 0.15

    for p in pop_sizes:
        print(f"\n--> Testing Population Size = {p}")
        ga.POP_SIZE = p
        fit, delay, energy, min_res, _ = ga.run_ga(seed=42)
        best_fits.append(fit)

    plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})
    plt.figure(figsize=(8, 6))
    plt.plot(pop_sizes, best_fits, marker="o", markersize=8, color="#17becf", linewidth=2.5)
    plt.xlabel("Population Size", fontweight='bold')
    plt.ylabel("Best Fitness Achieved", fontweight='bold')
    plt.title("Sensitivity Analysis: Effect of Population Size on Fitness", fontweight='bold', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.7)

    for i, txt in enumerate(best_fits):
        plt.annotate(f"{txt:.4f}", (pop_sizes[i], best_fits[i]), textcoords="offset points", xytext=(0, 10),
                     ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig("plot_6_population_sensitivity.png", dpi=300)
    print(" Population sensitivity plot saved as 'plot_6_population_sensitivity.png'")
    plt.show()

def test_crossover_sensitivity():
    print("\n" + "=" * 40)
    print(" Running Parameter Sensitivity: CROSSOVER RATE")
    print("=" * 40)

    crossover_rates = [0.6, 0.7, 0.75, 0.8, 0.9, 0.95]
    best_fits = []

    ga.POP_SIZE = 80
    ga.MUTATION_RATE = 0.15

    for c in crossover_rates:
        print(f"\n--> Testing Crossover Rate = {c}")
        ga.CROSSOVER_RATE = c
        fit, delay, energy, min_res, _ = ga.run_ga(seed=42)
        best_fits.append(fit)

    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})
    plt.figure(figsize=(8, 6))
    plt.plot(crossover_rates, best_fits, marker="^", markersize=8, color="#ff7f0e", linewidth=2.5)
    plt.xlabel("Crossover Rate (Probability)", fontweight='bold')
    plt.ylabel("Best Fitness Achieved", fontweight='bold')
    plt.title("Sensitivity Analysis: Effect of Crossover Rate on Fitness", fontweight='bold', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.7)

    for i, txt in enumerate(best_fits):
        plt.annotate(f"{txt:.4f}", (crossover_rates[i], best_fits[i]), textcoords="offset points", xytext=(0, 10),
                     ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig("plot_8_crossover_sensitivity.png", dpi=300)
    print(" Crossover sensitivity plot saved as 'plot_7_crossover_sensitivity.png'")
    plt.show()

if __name__ == "__main__":
    print("Starting Comprehensive Parameter Sensitivity Analysis...")
    test_mutation_sensitivity()
    test_population_sensitivity()
    test_crossover_sensitivity()
    print("\n ALL EXPERIMENTS AND PLOTS ARE COMPLETE! ")

