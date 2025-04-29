import os
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import NMF

os.makedirs('denoising_results', exist_ok=True)

original_image_path = 'gray_mandrill.jpg'
original_image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE) / 255.0
m, n = original_image.shape

noise_std = 0.1
np.random.seed(0)
noise = np.random.normal(0, noise_std**0.5, original_image.shape)
print(noise)
noisy_image = np.clip(original_image + noise,0,1)
plt.imsave('noisy_mandrill.jpg',noisy_image, cmap='gray')

plt.imsave('denoising_results/noisy_image.jpg', noisy_image, cmap='gray')

percentages=[0.01, 0.05, 0.1, 0.2]
solvers = ['mu', 'cd']
results = []
full_rank= np.linalg.matrix_rank(noisy_image)
print(' The matrix rank is ', full_rank)
for perc in percentages:
    rank=int(perc*full_rank)
    for solver in solvers:
        try:
            start_time = time.time()

            model = NMF(n_components=rank, init='random', solver=solver, max_iter=1500, random_state=42)
            W = model.fit_transform(noisy_image)
            H = model.components_
            reconstructed = np.clip(W @ H, 0, 1)

            end_time = time.time()
            duration = end_time - start_time

            mse = np.mean((original_image - reconstructed) ** 2)
            psnr = 10 * np.log10(1.0 / mse)

            img_filename = f'denoising_results/reconstructed_{perc}_{solver}.jpg'
            plt.imsave(img_filename, reconstructed, cmap='gray')

            results.append({
                'Solver': solver,
                'Rank': rank,
                'MSE': mse,
                'PSNR': psnr,
                'Time Taken (s)': duration
            })

            plt.figure(figsize=(4, 4))
            plt.imshow(reconstructed, cmap='gray')
            plt.title(f'{solver.upper()} | MSE: {mse:.5f} | PSNR: {psnr:.2f} dB | Time: {duration:.2f}s')
            plt.axis('off')
            plt.tight_layout()
        except Exception as e:
            print(f"Error with solver '{solver}': {e}")
            results.append({
                'Solver': solver,
                'Rank': rank,
                'MSE': None,
                'PSNR': None,
            '   Time Taken (s)': None
            })

    df = pd.DataFrame(results)
    df.to_csv('denoising_comparison_results.csv', index=False)
    print("Denoising comparison complete. Results saved to 'denoising_comparison_results.csv'.")
