import numpy as np
from scipy.optimize import nnls

# ---------- Gradient Descent NMF ----------
def gradient_descent(V, r, eta=0.001, max_iter=200, tol=1e-4, verbose=False):
    m, n = V.shape
    W = np.random.rand(m, r)
    H = np.random.rand(r, n)

    for t in range(1, max_iter + 1):
        V_hat = W @ H
        grad_W = (V_hat - V) @ H.T
        grad_H = W.T @ (V_hat - V)

        W -= eta * grad_W
        H -= eta * grad_H
        W = np.maximum(W, 0)
        H = np.maximum(H, 0)

        error = np.linalg.norm(V - V_hat, ord='fro')
        if verbose and t % 100 == 0:
            print(f"Iteration {t}: error = {error:.6f}")
        if error < tol:
            break

    return W, H


# ---------- Multiplicative Update NMF ----------
def multiplicative_update(V, k=50, max_iter=1000, epsilon=1e-10, tol=1e-4, verbose=True, random_state=42):
    np.random.seed(random_state)
    m, n = V.shape
    W = np.random.rand(m, k)
    H = np.random.rand(k, n)
    for t in range(max_iter):
        WH = W @ H
        H *= (W.T @ V) / (W.T @ WH + epsilon)
        W *= (V @ H.T) / (WH @ H.T + epsilon)
        V_hat = W @ H
        error = np.linalg.norm(V - V_hat, 'fro')
        if verbose and t % 100 == 0:
            print(f"[{t}] Reconstruction error: {error:.6f}")
        if error < tol:
            print(f"Converged at iteration {t} with error = {error:.6f}")
            break
    return W, H



# ---------- Alternating Least Squares (NNLS) ----------
def als(V, r, max_iter=200, epsilon=1e-4, verbose=False):
    m, n = V.shape
    print(f"Initializing ALS (NNLS) NMF with V shape: ({m}, {n}), rank: {r}")
    
    W = np.random.rand(m, r)
    H = np.random.rand(r, n)
    print("Randomly initialized W and H.")

    for t in range(1, max_iter + 1):
        print(f"[{t}] Solving NNLS for H columns...")
        for j in range(n):
            H[:, j], _ = nnls(W, V[:, j])

        print(f"[{t}] Solving NNLS for W rows...")
        for i in range(m):
            W[i, :], _ = nnls(H.T, V[i, :].T)

        V_hat = W @ H
        error = np.linalg.norm(V - V_hat, 'fro')
        print(f"[{t}] Reconstruction error: {error:.6f}")

        if verbose and t % 10 == 0:
            print(f"Verbose: Iteration {t}: error = {error:.6f}")

        if error < epsilon:
            print(f"Converged at iteration {t} with error = {error:.6f}")
            break

    return W, H
