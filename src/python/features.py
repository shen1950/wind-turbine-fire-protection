import numpy as np


def extract_stat_features(X):
    last = X[:, -1, :]
    mean = X.mean(axis=1)
    std = X.std(axis=1)
    mx = X.max(axis=1)
    slope = X[:, -1, :] - X[:, 0, :]
    recent_slope = X[:, -1, :] - X[:, max(0, X.shape[1] - 10), :]
    return np.concatenate([last, mean, std, mx, slope, recent_slope], axis=1)


class StandardScaler2D:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-8
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

