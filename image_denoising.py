import os
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from skimage import data, color, img_as_float
from skimage.util import random_noise
from skimage.metrics import structural_similarity
from sklearn.decomposition import NMF

os.makedirs('denoising_results', exist_ok=True)

original_image = color.rgb2gray(img_as_float(data.astronaut()))
plt.imsave('astronaut_original.jpg', original_image, cmap='gray')
m, n = original_image.shape

np.random.seed(0)
noisy_image = random_noise(original_image, mode='gaussian', var=0.01)
plt.imsave('denoising_results/noisy_image.jpg', noisy_image, cmap='gray')

percentages = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 0.90, 1]
solvers = ['mu', 'cd']
results = []
full_rank = np.linalg.matrix_rank(noisy_image)
print('The matrix rank is', full_rank)

for perc in percentages:
    rank = int(perc * full_rank)
    for solver in solvers:
        try:
            start_time = time.time()
            model = NMF(n_components=rank, init='random', solver=solver, max_iter=1500, random_state=42)
            W = model.fit_transform(noisy_image)
            H = model.components_
            V = np.dot(W, H)
            reconstructed = np.clip(V, 0, 1)
            end_time = time.time()
            duration = end_time - start_time
            mse = np.mean((original_image - reconstructed) ** 2)
            psnr = 10 * np.log10(1.0 / mse)
            ssim = structural_similarity(original_image, reconstructed, data_range=1.0)
            img_filename = f'denoising_results/reconstructed_{perc}_{solver}.jpg'
            plt.imsave(img_filename, reconstructed, cmap='gray')
            results.append({
                'Solver': solver,
                'Rank': rank,
                'Percentage': perc,
                'MSE': mse,
                'PSNR': psnr,
                'SSIM': ssim,
                'Time Taken (s)': duration
            })
            plt.figure(figsize=(4, 4))
            plt.imshow(reconstructed, cmap='gray')
            plt.title(f'{solver.upper()} | PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f}')
            plt.axis('off')
            plt.tight_layout()
        except Exception as e:
            print(f"Error with solver '{solver}': {e}")
            results.append({
                'Solver': solver,
                'Rank': rank,
                'Percentage': perc,
                'MSE': None,
                'PSNR': None,
                'SSIM': None,
                'Time Taken (s)': None
            })

df = pd.DataFrame(results)
df.to_csv('denoising_comparison_results.csv', index=False)
print("Denoising comparison complete. Results saved to 'denoising_comparison_results.csv'.")
