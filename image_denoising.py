import numpy as np
import matplotlib.pyplot as plt
from skimage import data, img_as_float
import os 
from skimage.util import random_noise
from skimage.color import rgb2gray
from nnmf import als, multiplicative_update  # Removed gradient_descent


img = data.camera() 
V = img_as_float(img)


V_noisy = random_noise(V, mode='gaussian', var=0.01)


rank = 50
T = 1000
tol = 1e-5
epsilon = 1e-9


print("Running Multiplicative Update NMF...")
W2, H2 = multiplicative_update(V_noisy, k=rank, max_iter=T, tol=tol, epsilon=epsilon)
V_denoised2 = W2 @ H2
V_denoised2_norm = (V_denoised2 - V_denoised2.min()) / (V_denoised2.max() - V_denoised2.min())
print(V_denoised2_norm)



print("Running NNLS-based NMF...")
W3, H3 = als(V_noisy, r=rank, max_iter=T, epsilon=tol)
V_denoised3 = np.clip(W3 @ H3, 0, 1)


fig, axes = plt.subplots(1, 4, figsize=(16, 4))
titles = ['Original', 'Noisy', 'Multiplicative', 'ALS']
images = [V, V_noisy, V_denoised2_norm, V_denoised3]

for ax, title, image in zip(axes, titles, images):
    ax.imshow(image, cmap='gray')
    ax.set_title(title)
    ax.axis('off')

save_dir = "result"
os.makedirs(save_dir, exist_ok=True)
output_path = os.path.join(save_dir, f'rank_{rank}_nmf_denoising_comparison.png')
plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches='tight')
