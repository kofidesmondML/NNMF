import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
import cv2
from skimage.color import rgb2gray
from skimage.io import imread
from skimage.transform import resize

mandrill_gray =cv2.imread('gray_mandrill.jpg', cv2.IMREAD_GRAYSCALE)

# Step 3: Apply NMF
# Note: grayscale image is shape (512, 512) → treat rows as samples
model = NMF(n_components=15, init='random', random_state=0)
W = model.fit_transform(mandrill_gray)
H = model.components_

# Reconstruct image
reconstructed = np.dot(W, H)

# Step 4: Visualize
plt.figure(figsize=(12, 6))

# Original image
plt.subplot(1, 3, 1)
plt.imshow(mandrill_gray, cmap='gray')
plt.title('Original Grayscale Image')
plt.axis('off')

# Reconstructed image
plt.subplot(1, 3, 2)
plt.imshow(reconstructed, cmap='gray')
plt.title('Reconstructed (NMF)')
plt.axis('off')

# Show first few basis components (columns of H)
plt.subplot(1, 3, 3)
for i in range(5):
    plt.plot(H[i], label=f'Component {i+1}')
plt.title('First 5 NMF Components (H)')
plt.legend()

plt.tight_layout()
plt.show()
