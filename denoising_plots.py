import pandas as pd
import matplotlib.pyplot as plt
import os

df_mul = pd.read_csv("denoising_comparison_results.csv")

metrics = ['MSE', 'PSNR', 'SSIM', 'Time Taken (s)']
colors = {'mu': 'blue', 'cd': 'orange'}

output_dir = "denoising_results"
os.makedirs(output_dir, exist_ok=True)

for metric in metrics:
    plt.figure(figsize=(8, 6))
    for solver in df_mul['Solver'].unique():
        subset = df_mul[df_mul['Solver'] == solver]
        plt.plot(subset['Rank'], subset[metric], marker='o', label=solver, color=colors.get(solver, None))
    plt.title(f'{metric} vs Rank')
    plt.xlabel('Rank')
    plt.ylabel(metric)
    plt.grid(True)
    plt.legend()
    filename = metric.lower().replace(" ", "_").replace("(", "").replace(")", "") + "_vs_rank_denoising.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300)
    plt.close()

