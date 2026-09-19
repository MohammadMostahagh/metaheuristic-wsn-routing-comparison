import csv
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def read_history_csv(filename):
    """تابع کمکی برای خواندن فایل‌های CSV و استخراج ستون‌ها به صورت لیست"""
    data = {
        "MaxFitness": [],
        "TotalEnergy_J": [],
        "AvgDelay_Hops": [],
        "MinResidual_J": []
    }

    try:
        with open(filename, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data["MaxFitness"].append(float(row["MaxFitness"]))
                data["TotalEnergy_J"].append(float(row["TotalEnergy_J"]))
                data["AvgDelay_Hops"].append(float(row["AvgDelay_Hops"]))
                data["MinResidual_J"].append(float(row["MinResidual_J"]))
        return data
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found! Please make sure it exists in the folder.")
        return None


def plot_algorithms_comparison():
    print("Reading data and generating comparison plots...")

    ga_data = read_history_csv("averaged_convergence_history.csv")
    pso_data = read_history_csv("pso_averaged_convergence_history.csv")

    if ga_data is None or pso_data is None:
        return

    iterations = range(len(ga_data["MaxFitness"]))

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Performance Comparison: Advanced GA vs. Discrete PSO (Average of 30 Runs)",
                 fontsize=16, fontweight='bold')

    axs[0, 0].plot(iterations, ga_data["MaxFitness"], label='Advanced GA', color='blue', linewidth=2)
    axs[0, 0].plot(iterations, pso_data["MaxFitness"], label='Discrete PSO', color='green', linewidth=2, linestyle='--')
    axs[0, 0].set_title('Fitness Convergence', fontweight='bold')
    axs[0, 0].set_xlabel('Generation / Iteration')
    axs[0, 0].set_ylabel('Best Fitness Value')
    axs[0, 0].grid(True, linestyle=':', alpha=0.6)
    axs[0, 0].legend()

    axs[0, 1].plot(iterations, ga_data["TotalEnergy_J"], label='Advanced GA', color='blue', linewidth=2)
    axs[0, 1].plot(iterations, pso_data["TotalEnergy_J"], label='Discrete PSO', color='green', linewidth=2,
                   linestyle='--')
    axs[0, 1].set_title('Optimization of Total Energy Consumption', fontweight='bold')
    axs[0, 1].set_xlabel('Generation / Iteration')
    axs[0, 1].set_ylabel('Energy (Joules)')
    axs[0, 1].grid(True, linestyle=':', alpha=0.6)
    axs[0, 1].legend()

    axs[1, 0].plot(iterations, ga_data["AvgDelay_Hops"], label='Advanced GA', color='blue', linewidth=2)
    axs[1, 0].plot(iterations, pso_data["AvgDelay_Hops"], label='Discrete PSO', color='green', linewidth=2,
                   linestyle='--')
    axs[1, 0].set_title('Optimization of Routing Delay', fontweight='bold')
    axs[1, 0].set_xlabel('Generation / Iteration')
    axs[1, 0].set_ylabel('Average Delay (Hops)')
    axs[1, 0].grid(True, linestyle=':', alpha=0.6)
    axs[1, 0].legend()

    axs[1, 1].plot(iterations, ga_data["MinResidual_J"], label='Advanced GA', color='blue', linewidth=2)
    axs[1, 1].plot(iterations, pso_data["MinResidual_J"], label='Discrete PSO', color='green', linewidth=2,
                   linestyle='--')
    axs[1, 1].set_title('Load Balancing (Protection of Bottleneck Nodes)', fontweight='bold')
    axs[1, 1].set_xlabel('Generation / Iteration')
    axs[1, 1].set_ylabel('Minimum Residual Energy (Joules)')
    axs[1, 1].grid(True, linestyle=':', alpha=0.6)
    axs[1, 1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("GA_vs_PSO_Comparison.png", dpi=300)
    print("Success! The comparison plot has been saved as 'GA_vs_PSO_Comparison.png'.")
    plt.show()


if __name__ == "__main__":
    plot_algorithms_comparison()