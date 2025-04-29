import os
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import NMF 

os.makedirs('results', exist_ok=True)

original_image_path = 'gray_mandrill.jpg'
original_file_size = os.path.getsize(original_image_path)

gray_image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE) / 255.0

full_rank = np.linalg.matrix_rank(gray_image)
percentages = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 0.90, 1]

m, n = gray_image.shape

columns = ['Percentage of Rank', 'Rank', 'Compression Ratio', 'MSE', 'PSNR', 'Time Taken (s)']
results_df = pd.DataFrame(columns=columns)

for idx, perc in enumerate(percentages):
    rank = max(1, int(perc * full_rank))
    try:
        start_time = time.time()

        model = NMF(n_components=rank, init='random', solver='mu', max_iter=1500, random_state=42)
        W = model.fit_transform(gray_image)
        H = model.components_

        end_time = time.time()
        comp_time = end_time - start_time

        compressed_img = W @ H

        compressed_image_path = f'results/mu_compressed_rank_{rank}_perc_{int(perc * 100)}.jpg'
        plt.imsave(compressed_image_path, compressed_img, cmap='gray')

        compressed_file_size = os.path.getsize(compressed_image_path)
        real_compression_ratio = compressed_file_size / original_file_size

        mse = np.linalg.norm(gray_image - compressed_img, 'fro')**2 / (m * n)
        psnr = 10 * np.log10(1.0 / mse)

        new_row = pd.DataFrame({
            'Percentage of Rank': [perc],
            'Rank': [rank],
            'Compression Ratio': [real_compression_ratio],
            'MSE': [mse],
            'PSNR': [psnr],
            'Time Taken (s)': [comp_time]
        })
        results_df = pd.concat([results_df, new_row], ignore_index=True)

        plt.figure(figsize=(4, 4))
        plt.title(
            f'Rank ≈ {rank} ({int(perc * 100)}%)\n'
            f'Time: {comp_time:.2f}s\n'
            f'Real Compression Ratio: {real_compression_ratio:.4f}\n'
            f'MSE: {mse:.6f}\n'
            f'PSNR: {psnr:.2f} dB'
        )
        plt.imshow(compressed_img, cmap='gray')
        plt.axis('off')

    except Exception as e:
        print(f"Error processing rank {rank} ({perc*100}%): {e}")
        error_message = str(e)
        new_row = pd.DataFrame({
            'Percentage of Rank': [perc],
            'Rank': [rank],
            'Compression Ratio': [None],
            'MSE': [None],
            'PSNR': [None],
            'Time Taken (s)': [None]
        })
        results_df = pd.concat([results_df, new_row], ignore_index=True)
        with open("compression_errors.log", "a") as log_file:
            log_file.write(f"Error processing rank {rank} ({perc*100}%): {error_message}\n")

results_df.to_csv('multiplicative_updates_results.csv', index=False)
print("Processing complete. Results saved to 'multiplicative_updates_results.csv'.")
