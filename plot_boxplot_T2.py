import pandas as pd
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def plot_boxplots():
    print("Generating Robustness Boxplots...")

    try:
        ga_data = pd.read_csv("multi_run_summary.csv")
        pso_data = pd.read_csv("pso_multi_run_summary.csv")
    except FileNotFoundError:
        print("Error: Summary CSV files not found!")
        return

    ga_fit, pso_fit = ga_data["Best_Fitness"], pso_data["Best_Fitness"]
    ga_eng, pso_eng = ga_data["Best_Energy_J"], pso_data["Best_Energy_J"]
    ga_del, pso_del = ga_data["Best_Delay_Hops"], pso_data["Best_Delay_Hops"]

    fig, axs = plt.subplots(1, 3, figsize=(15, 6))
    fig.suptitle("Algorithm Stability & Robustness Over 30 Independent Runs\nAdvanced GA vs Discrete PSO",
                 fontsize=14, fontweight='bold')

    box_colors = ['lightblue', 'lightgreen']
    labels = ['Advanced GA', 'Discrete PSO']

    # 1. Boxplot فیتنس
    bplot1 = axs[0].boxplot([ga_fit, pso_fit], labels=labels, patch_artist=True)
    axs[0].set_title('Fitness Stability', fontweight='bold')
    axs[0].set_ylabel('Best Fitness Value')
    axs[0].grid(True, linestyle=':', alpha=0.6)

    # 2. Boxplot انرژی
    bplot2 = axs[1].boxplot([ga_eng, pso_eng], labels=labels, patch_artist=True)
    axs[1].set_title('Energy Consumption Stability', fontweight='bold')
    axs[1].set_ylabel('Total Energy (Joules)')
    axs[1].grid(True, linestyle=':', alpha=0.6)

    # 3. Boxplot تاخیر
    bplot3 = axs[2].boxplot([ga_del, pso_del], labels=labels, patch_artist=True)
    axs[2].set_title('Routing Delay Stability', fontweight='bold')
    axs[2].set_ylabel('Average Delay (Hops)')
    axs[2].grid(True, linestyle=':', alpha=0.6)

    for bplot in (bplot1, bplot2, bplot3):
        for patch, color in zip(bplot['boxes'], box_colors):
            patch.set_facecolor(color)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("Robustness_Boxplot_Comparison.png", dpi=300)
    print("Success! Boxplot saved as 'Robustness_Boxplot_Comparison.png'")
    plt.show()


if __name__ == "__main__":
    plot_boxplots()