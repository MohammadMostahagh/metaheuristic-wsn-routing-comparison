import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def generate_statistical_report():
    print("=" * 50)
    print("  STATISTICAL ROBUSTNESS REPORT (Advanced GA) ")
    print("=" * 50)

    try:
        df = pd.read_csv("multi_run_summary.csv")
    except FileNotFoundError:
        print("Error: 'multi_run_summary.csv' not found. Run main.py first.")
        return

    fitness = df["Best_Fitness"]
    energy = df["Best_Energy_J"]
    delay = df["Best_Delay_Hops"]

    metrics = {
        "Best Fitness": fitness,
        "Total Energy (Joules)": energy,
        "Average Delay (Hops)": delay
    }

    for name, data in metrics.items():
        mean_val = np.mean(data)
        std_val = np.std(data)
        min_val = np.min(data)
        max_val = np.max(data)

        print(f"\n {name}:")
        print(f"   Mean (میانگین): {mean_val:.6f}")
        print(f"   Std. Dev (انحراف معیار): {std_val:.6f}")
        print(f"   Best (بهترین): {max_val if name == 'Best Fitness' else min_val:.6f}")
        print(f"   Worst (بدترین): {min_val if name == 'Best Fitness' else max_val:.6f}")

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({'font.size': 11, 'font.family': 'sans-serif'})

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Algorithm Stability & Robustness Over 5 Independent Runs", fontweight='bold', fontsize=14, y=1.02)

    boxprops = dict(facecolor='#aec7e8', color='#1f77b4', linewidth=2)
    medianprops = dict(color='#d62728', linewidth=2)
    whiskerprops = dict(color='#1f77b4', linewidth=2)
    capprops = dict(color='#1f77b4', linewidth=2)

    # 1. Fitness Boxplot
    axes[0].boxplot([fitness], tick_labels=["Advanced GA"], patch_artist=True,
                    boxprops=boxprops, medianprops=medianprops, whiskerprops=whiskerprops, capprops=capprops)
    axes[0].set_title("Fitness Stability", fontweight='bold')
    axes[0].set_ylabel("Best Fitness Value")

    # 2. Energy Boxplot
    axes[1].boxplot([energy], tick_labels=["Advanced GA"], patch_artist=True,
                    boxprops=boxprops, medianprops=medianprops, whiskerprops=whiskerprops, capprops=capprops)
    axes[1].set_title("Energy Consumption Stability", fontweight='bold')
    axes[1].set_ylabel("Total Energy (Joules)")

    # 3. Delay Boxplot
    axes[2].boxplot([delay], tick_labels=["Advanced GA"], patch_artist=True,
                    boxprops=boxprops, medianprops=medianprops, whiskerprops=whiskerprops, capprops=capprops)
    axes[2].set_title("Routing Delay Stability", fontweight='bold')
    axes[2].set_ylabel("Average Delay (Hops)")

    plt.tight_layout()
    plt.savefig("plot_7_robustness_boxplot.png", dpi=300, bbox_inches='tight')
    print("\n Boxplot saved successfully as 'plot_7_robustness_boxplot.png'")
    plt.show()


if __name__ == "__main__":
    generate_statistical_report()