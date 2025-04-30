import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import NMF
from tensorflow.keras.datasets import fashion_mnist

save_dir = "feature_extraction"
os.makedirs(save_dir, exist_ok=True)

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X_all = np.concatenate((X_train, X_test), axis=0)
y_all = np.concatenate((y_train, y_test), axis=0)

trouser_indices = np.where(y_all == 1)[0]
X_trousers = X_all[trouser_indices][:6000]

plt.figure(figsize=(10, 2))
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(X_trousers[i], cmap='gray')
    plt.axis('off')
plt.suptitle("Sample Trouser Images (Label 1)")
plt.tight_layout()
plt.savefig(f"{save_dir}/sample_trousers.png")
plt.show()

A = X_trousers.reshape(6000, 28*28).T.astype(np.float32) / 255.0 

nmf_model = NMF(n_components=12, init='random', random_state=0, max_iter=970)
W = nmf_model.fit_transform(A)
H = nmf_model.components_

np.save(f"{save_dir}/W.npy", W)
np.save(f"{save_dir}/H.npy", H)

print("W shape:", W.shape)
print("H shape:", H.shape)

plt.figure(figsize=(10, 4))
for i in range(12):
    component_image = W[:, i].reshape(28, 28)
    plt.subplot(2, 6, i + 1)
    plt.imshow(component_image, cmap='gray')
    plt.title(f"Feature {i+1}")
    plt.axis('off')
    plt.imsave(f"{save_dir}/Feature_{i+1}.png", component_image, cmap='gray')

plt.suptitle("First 12 NMF Components (Trousers)")
plt.tight_layout()
plt.savefig(f"{save_dir}/nmf_components_grid.png")