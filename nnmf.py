import numpy as np
from scipy.optimize import nnls 

def gradient_descent(V, r, eta=0.001, max_iter=1000, tol=1e-4, verbose=False):
    m, n = V.shape
    W = np.random.rand(m, r)
    H = np.random.rand(r, n)
    for t in range(1, max_iter + 1):
        V_hat = np.dot(W, H)
        grad_W = (V + V_hat).dot(H.T)
        grad_H = W.T.dot(V + V_hat)
        W -= eta * grad_W
        H -= eta * grad_H
        W = np.maximum(W, 0)
        H = np.maximum(H, 0)
        error = np.linalg.norm(V - np.dot(W, H), ord='fro')
        if verbose and t % 100 == 0:
            print(f"Iteration {t}: Reconstruction error = {error:.6f}")

        if error < tol:
            if verbose:
                print(f"Converged at iteration {t} with error = {error:.6f}")
            break
    return W, H




def multiplicative_update(V, r, T=1000, tol=1e-4, epsilon=1e-10, verbose=False):
    m, n = V.shape
    W = np.random.rand(m, r)
    H = np.random.rand(r, n)
    for t in range(1, T + 1):
        WH = W @ H
        numerator_H = W.T @ V
        denominator_H = W.T @ WH + epsilon
        H *= numerator_H / denominator_H
        WH = W @ H
        numerator_W = V @ H.T
        denominator_W = W @ (H @ H.T) + epsilon
        W *= numerator_W / denominator_W
        V_hat = W @ H
        error = np.linalg.norm(V - V_hat, 'fro')
        if verbose and t % 100 == 0:
            print(f"Iteration {t}: error = {error:.6f}")

        if error < tol:
            if verbose:
                print(f"Converged at iteration {t} with error = {error:.6f}")
            break
    return W, H

import numpy as np
from scipy.optimize import nnls

def als(V, r, T=100, epsilon=1e-4, verbose=False):
    m, n = V.shape
    W = np.random.rand(m, r)
    H = np.random.rand(r, n)

    for t in range(1, T + 1):
        for j in range(n):
            H[:, j], _ = nnls(W, V[:, j])
        for i in range(m):
            W[i, :], _ = nnls(H.T, V[i, :].T)

        V_hat = W @ H
        error = np.linalg.norm(V - V_hat, 'fro')

        if verbose and t % 10 == 0:
            print(f"Iteration {t}: Reconstruction error = {error:.6f}")
        if error < epsilon:
            if verbose:
                print(f"Converged at iteration {t} with error = {error:.6f}")
            break

    return W, H


