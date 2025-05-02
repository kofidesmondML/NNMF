import pandas as pd
import matplotlib.pyplot as plt

df_mul = pd.read_csv("multiplicative_updates_results.csv")
df_als = pd.read_csv("als_results.csv")

df_mul.columns = df_mul.columns.str.strip()
df_als.columns = df_als.columns.str.strip()

def plot_comparison(x, y, y_label, title, filename):
    file_name='results/'+filename
    plt.figure(figsize=(8, 5))
    plt.plot(df_mul[x], df_mul[y], 'o-', label='Multiplicative Updates')
    plt.plot(df_als[x], df_als[y], 's--', label='Alternating Least Squares')
    plt.xlabel(x)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{file_name}.png")

plot_comparison("Rank", "Compression Ratio", "Compression Ratio", "Compression Ratio vs Rank", "compression_ratio")
plot_comparison("Rank", "MSE", "Mean Squared Error (MSE)", "MSE vs Rank", "mse_vs_rank")
plot_comparison("Rank", "PSNR", "Peak Signal-to-Noise Ratio (PSNR)", "PSNR vs Rank", "psnr_vs_rank")
plot_comparison("Rank", "Time Taken (s)", "Time Taken (s)", "Time Taken vs Rank", "time_vs_rank")
