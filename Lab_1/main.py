import numpy as np
from matplotlib import pyplot as plt

from numpy.linalg import inv
from tqdm import trange


def l2_error(X, t, w):
    return np.sum((X.dot(w.ravel()) - t) ** 2)


def get_w_sigma(X, t, alpha, beta):
    """Calculate the mean and the covariance matrix
       of the posterior distribution"""
    n, d = X.shape

    sigma = inv(beta * X.T @ X + alpha)
    w = beta * sigma @ X.T @ t

    w[w < 1e-10] = 0

    return w, sigma


def update_alpha_beta(X, t, alpha, beta):
    """Update the hyperperemeters to increase evidence"""
    w, sigma = get_w_sigma(X, t, alpha, beta)

    alpha_new = (1-np.diag(alpha)@np.diag(sigma))/(w*w)
    beta_new = (len(X) - (1 - alpha * sigma).sum()) / np.linalg.norm(t - X @ w)

    return alpha_new, beta_new


def fit_rvr(X, t, max_iter=10000):
    """Train the Relevance Vector Regression model"""

    alpha = np.ones(X.shape[1])
    beta = 1
    for i in trange(max_iter):
        alpha, beta = update_alpha_beta(X, t, alpha, beta)
    w, sigma = get_w_sigma(X, t, alpha, beta)

    return w, sigma, alpha, beta


# Data generation

def gen_batch(n, w, beta):
    d = len(w)
    X = np.random.uniform(-1, 1, (n, 1))
    X = np.sort(X, axis=0)
    X = np.hstack([X ** i for i in range(d)])
    t = X.dot(w) + np.random.normal(size=n) / beta ** 0.5
    return X, t


n = 200
d = 21
w_true = np.zeros(d)
w_true[1] = 1
w_true[3] = -1
beta_true = 100

X_train, t_train = gen_batch(n, w_true, beta_true)
X_test, t_test = gen_batch(n, w_true, beta_true)

# # Visualization
# fig, ax = plt.subplots()
# ax.scatter(X_train[:, 1], t_train, s=3, label='Train data', alpha=0.3)
# ax.scatter(X_test[:, 1], t_test, s=3, label='Test data', alpha=0.3)
# ax.plot(X_train[:, 1], X_train.dot(w_true), label='Ground truth')
#
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.legend(ncol=3, loc=9, bbox_to_anchor=(0.5, 1.15))
# plt.show()

# Relevance Vector Regression
w_rvr, sigma_rvr, alpha_rvr, beta_rvr = fit_rvr(X_train, t_train)
