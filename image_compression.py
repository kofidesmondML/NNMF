import matplotlib.pyplot as plt 
import cv2 
from PIL import Image 
import time
import os 
import numpy as np
from nnmf import multiplicative_update, als



image=cv2.imread('alone.jpg')

#image.show()

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#gray_image.show()

cv2.imwrite('gray_alone.jpg', gray_image)

image = cv2.imread('gray_alone.jpg', cv2.IMREAD_GRAYSCALE)
original_shape = image.shape
original_size = image.size
print(f"Loaded image with shape {original_shape} and total size {original_size} pixels.")

print(">>> Computing full rank of the image...")
full_rank = np.linalg.matrix_rank(image)
print(f"The full rank of the image is: {full_rank}")

percentages = [1, 5, 10, 20, 30, 50, 79, 85, 90, 100]
print(f"Compression will be tested for the following percentages of the rank: {percentages}")

print(">>> Checking for 'results' folder...")
os.makedirs('results', exist_ok=True)
print("Folder 'results/' is ready.")

results = []

for perc in percentages:
    print(f"\n=== Processing {perc}% of the rank ===")
    r = max(1, int(full_rank * perc / 100))
    print(f"Target rank: {r}")

    print(f">>> Multiplicative Update: Compressing with rank {r}...")
    start_time = time.time()
    W_mu, H_mu = multiplicative_update(image, r, max_iter=200)
    end_time = time.time()
    time_mu = end_time - start_time
    print(f"Multiplicative Update done in {time_mu:.4f} seconds.")

    compressed_image_mu = np.dot(W_mu, H_mu)
    compressed_image_mu = np.clip(compressed_image_mu, 0, 255).astype(np.uint8)
    compressed_size_mu = W_mu.size + H_mu.size

    compression_ratio_mu = compressed_size_mu / original_size
    frobenius_mu = np.linalg.norm(image - compressed_image_mu, 'fro')
    print(f"Multiplicative Update: Compression ratio = {compression_ratio_mu:.4f}, Frobenius norm = {frobenius_mu:.4f}")

    filename_mu = f'results/compressed_mu_{perc}percent.jpg'
    cv2.imwrite(filename_mu, compressed_image_mu)
    print(f"Multiplicative Update: Compressed image saved as {filename_mu}")

    print(f">>> ALS: Compressing with rank {r}...")
    start_time = time.time()
    W_als, H_als = als(image, r, max_iter=200)
    end_time = time.time()
    time_als = end_time - start_time
    print(f"ALS done in {time_als:.4f} seconds.")

    compressed_image_als = np.dot(W_als, H_als)
    compressed_image_als = np.clip(compressed_image_als, 0, 255).astype(np.uint8)
    compressed_size_als = W_als.size + H_als.size

    compression_ratio_als = compressed_size_als / original_size
    frobenius_als = np.linalg.norm(image - compressed_image_als, 'fro')
    print(f"ALS: Compression ratio = {compression_ratio_als:.4f}, Frobenius norm = {frobenius_als:.4f}")

    filename_als = f'results/compressed_als_{perc}percent.jpg'
    cv2.imwrite(filename_als, compressed_image_als)
    print(f"ALS: Compressed image saved as {filename_als}")

    results.append({
        'rank': r,
        'percentage': perc,
        'multiplicative_update': {
            'time': time_mu,
            'compression_ratio': compression_ratio_mu,
            'frobenius_norm': frobenius_mu,
            'saved_image': filename_mu
        },
        'als': {
            'time': time_als,
            'compression_ratio': compression_ratio_als,
            'frobenius_norm': frobenius_als,
            'saved_image': filename_als
        }
    })

print("\n>>> Summary of Results:")
for res in results:
    print(f"\n--- Rank {res['rank']} ({res['percentage']}%) ---")
    print(f"Multiplicative Update:")
    print(f"  Time: {res['multiplicative_update']['time']:.4f}s")
    print(f"  Compression Ratio: {res['multiplicative_update']['compression_ratio']:.4f}")
    print(f"  Frobenius Norm: {res['multiplicative_update']['frobenius_norm']:.4f}")
    print(f"  Image Saved: {res['multiplicative_update']['saved_image']}")
    
    print(f"ALS:")
    print(f"  Time: {res['als']['time']:.4f}s")
    print(f"  Compression Ratio: {res['als']['compression_ratio']:.4f}")
    print(f"  Frobenius Norm: {res['als']['frobenius_norm']:.4f}")
    print(f"  Image Saved: {res['als']['saved_image']}")

print("\n>>> All compressions completed successfully!")