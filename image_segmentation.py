import numpy as np
import matplotlib.pyplot as plt
from skimage import data, img_as_float
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from nnmf import multiplicative_update, als  # Ensure this returns W, H with internal init

# Load and preprocess image
img = data.camera()
V = img_as_float(img)
m, n = V.shape
print(m,n)

# Reshape the image to matrix form for NMF
V_reshaped = V.reshape((m, n))
#print(V.shape)

# Apply NMF
rank = 10  # Number of segments/components
W, H = multiplicative_update(V, k=rank, max_iter=200)

# Segment the image using KMeans on the W matrix
# Each row of W represents the membership of a pixel to a component
W_norm = normalize(W, axis=1)
kmeans = KMeans(n_clusters=rank, random_state=0)
labels = kmeans.fit_predict(W_norm)

# Reshape segmentation result to original image shape
segmented_image = labels.reshape(m, 1).repeat(n, axis=1)

# Display segmentation
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(V, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Segmented Image (NMF + KMeans)")
plt.imshow(segmented_image, cmap='gray')  # categorical colormap
plt.axis('off')
plt.tight_layout()
plt.show()
