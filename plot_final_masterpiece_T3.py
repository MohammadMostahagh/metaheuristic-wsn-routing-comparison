import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def plot_final_masterpiece():
    print("Reading data from all 3 algorithms (GA, PSO, SA)...")
    try:
        ga_hist = pd.read_csv("averaged_convergence_history.csv")
        pso_hist = pd.read_csv("pso_averaged_convergence_history.csv")
        sa_hist = pd.read_csv("sa_averaged_convergence_history.csv")

        ga_sum = pd.read_csv("multi_run_summary.csv")
        pso_sum = pd.read_csv("pso_multi_run_summary.csv")
        sa_sum = pd.read_csv("sa_multi_run_summary.csv")
    except FileNotFoundError as e:
        print(f"Error loading files. Ensure all CSVs are in the folder.\nDetails: {e}")
        return

    iterations = range(len(ga_hist))

    # ==========================================
    # ==========================================
    fig1, axs = plt.subplots(2, 2, figsize=(15, 11))
    fig1.suptitle("Final Battle: Advanced GA vs Discrete PSO vs SA (Average of 30 Runs)",
                  fontsize=16, fontweight='bold')

    # 1. همگرایی فیتنس
    axs[0, 0].plot(iterations, ga_hist["MaxFitness"], label='Advanced GA', color='blue', linewidth=2.5)
    axs[0, 0].plot(iterations, pso_hist["MaxFitness"], label='Discrete PSO', color='green', linewidth=2, linestyle='--')
    axs[0, 0].plot(iterations, sa_hist["MaxFitness"], label='Simulated Annealing', color='red', linewidth=2, linestyle='-.')
    axs[0, 0].set_title('Fitness Convergence', fontweight='bold')
    axs[0, 0].set_xlabel('Iteration / Generation / Temp Step')
    axs[0, 0].set_ylabel('Best Fitness Value')
    axs[0, 0].grid(True, linestyle=':', alpha=0.6)
    axs[0, 0].legend()

    # 2. مصرف انرژی
    axs[0, 1].plot(iterations, ga_hist["TotalEnergy_J"], label='Advanced GA', color='blue', linewidth=2.5)
    axs[0, 1].plot(iterations, pso_hist["TotalEnergy_J"], label='Discrete PSO', color='green', linewidth=2, linestyle='--')
    axs[0, 1].plot(iterations, sa_hist["TotalEnergy_J"], label='Simulated Annealing', color='red', linewidth=2, linestyle='-.')
    axs[0, 1].set_title('Total Energy Consumption', fontweight='bold')
    axs[0, 1].set_xlabel('Iteration / Generation / Temp Step')
    axs[0, 1].set_ylabel('Energy (Joules)')
    axs[0, 1].grid(True, linestyle=':', alpha=0.6)
    axs[0, 1].legend()

    # 3. تاخیر مسیر
    axs[1, 0].plot(iterations, ga_hist["AvgDelay_Hops"], label='Advanced GA', color='blue', linewidth=2.5)
    axs[1, 0].plot(iterations, pso_hist["AvgDelay_Hops"], label='Discrete PSO', color='green', linewidth=2, linestyle='--')
    axs[1, 0].plot(iterations, sa_hist["AvgDelay_Hops"], label='Simulated Annealing', color='red', linewidth=2, linestyle='-.')
    axs[1, 0].set_title('Average Routing Delay', fontweight='bold')
    axs[1, 0].set_xlabel('Iteration / Generation / Temp Step')
    axs[1, 0].set_ylabel('Delay (Hops)')
    axs[1, 0].grid(True, linestyle=':', alpha=0.6)
    axs[1, 0].legend()

    # 4. توازن بار (انرژی باقیمانده)
    axs[1, 1].plot(iterations, ga_hist["MinResidual_J"], label='Advanced GA', color='blue', linewidth=2.5)
    axs[1, 1].plot(iterations, pso_hist["MinResidual_J"], label='Discrete PSO', color='green', linewidth=2, linestyle='--')
    axs[1, 1].plot(iterations, sa_hist["MinResidual_J"], label='Simulated Annealing', color='red', linewidth=2, linestyle='-.')
    axs[1, 1].set_title('Load Balancing (Min Residual Energy)', fontweight='bold')
    axs[1, 1].set_xlabel('Iteration / Generation / Temp Step')
    axs[1, 1].set_ylabel('Residual Energy (Joules)')
    axs[1, 1].grid(True, linestyle=':', alpha=0.6)
    axs[1, 1].legend()

    fig1.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig1.savefig("Final_Convergence_Comparison.png", dpi=300)

    # ==========================================
    # ==========================================
    fig2, bxs = plt.subplots(1, 3, figsize=(16, 6))
    fig2.suptitle("Algorithm Stability & Robustness Over 30 Independent Runs", fontsize=15, fontweight='bold')

    labels = ['Adv. GA', 'Disc. PSO', 'SA']
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
    fig2.savefig("Final_Robustness_Boxplots.png", dpi=300)

    print("\nSUCCESS! Final masterpiece plots saved in your project folder:")
    print(" 1. Final_Convergence_Comparison.png")
    print(" 2. Final_Robustness_Boxplots.png")
    plt.show()

if __name__ == "__main__":
    plot_final_masterpiece()