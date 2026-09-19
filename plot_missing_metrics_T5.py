import pandas as pd
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np


def plot_missing_metrics():
    print("Reading data for Average Fitness and Diversity...")
    try:
        ga_hist = pd.read_csv("averaged_convergence_history.csv")
        pso_hist = pd.read_csv("pso_averaged_convergence_history.csv")
        sa_hist = pd.read_csv("sa_averaged_convergence_history.csv")
    except FileNotFoundError as e:
        print(f"Error loading files. Ensure GA, PSO, and SA CSVs are in the folder.\nDetails: {e}")
        return

    col_avg = "AvgFitness" if "AvgFitness" in ga_hist.columns else "AverageFitness"
    col_max = "MaxFitness" if "MaxFitness" in ga_hist.columns else "BestFitness"

    # ==========================================
    # ==========================================
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Population Dynamics: Average Fitness & Diversity", fontsize=16, fontweight='bold')

    lines = [
        (ga_hist, 'Advanced GA', 'blue', '-', 2.5),
        (pso_hist, 'Discrete PSO', 'green', '--', 2),
        (sa_hist, 'Standard SA', 'red', '-.', 2)
    ]

    for hist, label, color, style, width in lines:
        iters = range(len(hist))

        axs[0].plot(iters, hist[col_avg], label=label, color=color, linestyle=style, linewidth=width)

        diversity = hist[col_max] - hist[col_avg]
        diversity = np.maximum(diversity, 0)

        axs[1].plot(iters, diversity, label=label, color=color, linestyle=style, linewidth=width)

    axs[0].set_title('Average Population Fitness', fontweight='bold')
    axs[0].set_xlabel('Iteration / Generation / Temp Step')
    axs[0].set_ylabel('Average Fitness Value')
    axs[0].grid(True, linestyle=':', alpha=0.6)
    axs[0].legend()

    axs[1].set_title('Population Diversity (Max - Average Gap)', fontweight='bold')
    axs[1].set_xlabel('Iteration / Generation / Temp Step')
    axs[1].set_ylabel('Diversity Gap')
    axs[1].grid(True, linestyle=':', alpha=0.6)
    axs[1].legend()

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig("Missing_Metrics_Avg_and_Diversity.png", dpi=300)
    print("\nSUCCESS! Missing plots (Average Fitness & Diversity) saved.")
    plt.show()


if __name__ == "__main__":
    plot_missing_metrics()