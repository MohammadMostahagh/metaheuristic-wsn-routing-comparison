import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def plot_final_masterpiece():
    print("Reading data from 3 core algorithms (GA, PSO, Standard SA)...")
    try:
        ga_hist = pd.read_csv("averaged_convergence_history.csv")
        pso_hist = pd.read_csv("pso_averaged_convergence_history.csv")
        sa_hist = pd.read_csv("sa_averaged_convergence_history.csv")

        ga_sum = pd.read_csv("multi_run_summary.csv")
        pso_sum = pd.read_csv("pso_multi_run_summary.csv")
        sa_sum = pd.read_csv("sa_multi_run_summary.csv")
    except FileNotFoundError as e:
        print(f"Error loading files. Ensure GA, PSO, and SA CSVs are in the folder.\nDetails: {e}")
        return

    # ==========================================
    # ==========================================
    fig1, axs = plt.subplots(2, 2, figsize=(15, 11))
    fig1.suptitle("Final Comparison: Advanced GA vs Discrete PSO vs Standard SA",
                  fontsize=17, fontweight='bold')

    lines = [
        (ga_hist, 'Advanced GA', 'blue', '-', 2.5),
        (pso_hist, 'Discrete PSO', 'green', '--', 2),
        (sa_hist, 'Standard SA', 'red', '-.', 2)
    ]

    for hist, label, color, style, width in lines:
        iters = range(len(hist))  # طول محور X مخصوص همین فایل
        axs[0, 0].plot(iters, hist["MaxFitness"], label=label, color=color, linestyle=style, linewidth=width)
        axs[0, 1].plot(iters, hist["TotalEnergy_J"], label=label, color=color, linestyle=style, linewidth=width)
        axs[1, 0].plot(iters, hist["AvgDelay_Hops"], label=label, color=color, linestyle=style, linewidth=width)
        axs[1, 1].plot(iters, hist["MinResidual_J"], label=label, color=color, linestyle=style, linewidth=width)

    axs[0, 0].set_title('Fitness Convergence', fontweight='bold')
    axs[0, 0].set_xlabel('Iteration / Generation / Temp Step')
    axs[0, 0].set_ylabel('Best Fitness Value')
    axs[0, 0].grid(True, linestyle=':', alpha=0.6)
    axs[0, 0].legend()

    axs[0, 1].set_title('Total Energy Consumption', fontweight='bold')
    axs[0, 1].set_xlabel('Iteration / Generation / Temp Step')
    axs[0, 1].set_ylabel('Energy (Joules)')
    axs[0, 1].grid(True, linestyle=':', alpha=0.6)
    axs[0, 1].legend()

    axs[1, 0].set_title('Average Routing Delay', fontweight='bold')
    axs[1, 0].set_xlabel('Iteration / Generation / Temp Step')
    axs[1, 0].set_ylabel('Delay (Hops)')
    axs[1, 0].grid(True, linestyle=':', alpha=0.6)
    axs[1, 0].legend()

    axs[1, 1].set_title('Load Balancing (Min Residual Energy)', fontweight='bold')
    axs[1, 1].set_xlabel('Iteration / Generation / Temp Step')
    axs[1, 1].set_ylabel('Residual Energy (Joules)')
    axs[1, 1].grid(True, linestyle=':', alpha=0.6)
    axs[1, 1].legend()

    fig1.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig1.savefig("Final_Convergence_3Algos.png", dpi=300)

    # ==========================================
    # ==========================================
    fig2, bxs = plt.subplots(1, 3, figsize=(16, 6))
    fig2.suptitle("Algorithm Stability & Robustness Over 30 Independent Runs", fontsize=16, fontweight='bold')

    labels = ['Advanced GA', 'Discrete PSO', 'Standard SA']
    colors = ['lightblue', 'lightgreen', 'lightcoral']

    # Boxplot فیتنس
    bp1 = bxs[0].boxplot([ga_sum["Best_Fitness"], pso_sum["Best_Fitness"], sa_sum["Best_Fitness"]],
                         tick_labels=labels, patch_artist=True)
    bxs[0].set_title('Fitness Stability', fontweight='bold')
    bxs[0].set_ylabel('Best Fitness Value')
    bxs[0].grid(True, linestyle=':', alpha=0.6)

    # Boxplot انرژی
    bp2 = bxs[1].boxplot([ga_sum["Best_Energy_J"], pso_sum["Best_Energy_J"], sa_sum["Best_Energy_J"]],
                         tick_labels=labels, patch_artist=True)
    bxs[1].set_title('Energy Consumption Stability', fontweight='bold')
    bxs[1].set_ylabel('Total Energy (Joules)')
    bxs[1].grid(True, linestyle=':', alpha=0.6)

    # Boxplot تاخیر
    bp3 = bxs[2].boxplot([ga_sum["Best_Delay_Hops"], pso_sum["Best_Delay_Hops"], sa_sum["Best_Delay_Hops"]],
                         tick_labels=labels, patch_artist=True)
    bxs[2].set_title('Routing Delay Stability', fontweight='bold')
    bxs[2].set_ylabel('Average Delay (Hops)')
    bxs[2].grid(True, linestyle=':', alpha=0.6)

    for bp in (bp1, bp2, bp3):
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

    fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig2.savefig("Final_Robustness_Boxplots_3Algos.png", dpi=300)

    print("\nSUCCESS! Final 3-Algorithm plots saved.")
    plt.show()

if __name__ == "__main__":
    plot_final_masterpiece()