import numpy as np


class MinMaxScaler3D:
    def __init__(self):
        self.min_ = None
        self.max_ = None

    def fit(self, X):
        self.min_ = X.reshape(-1, X.shape[-1]).min(axis=0)
        self.max_ = X.reshape(-1, X.shape[-1]).max(axis=0)
        return self

    def transform(self, X):
        denom = np.maximum(self.max_ - self.min_, 1e-8)
        return np.clip((X - self.min_) / denom, 0.0, 1.0)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def smooth_sequence(X, window=3):
    if window <= 1:
        return X.copy()
    pad = window // 2
    padded = np.pad(X, ((0, 0), (pad, pad), (0, 0)), mode="edge")
    out = np.zeros_like(X)
    for i in range(X.shape[1]):
        out[:, i, :] = padded[:, i:i + window, :].mean(axis=1)
    return out

