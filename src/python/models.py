import numpy as np

from features import StandardScaler2D, extract_stat_features


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


class ThresholdRiskModel:
    def __init__(self, threshold=0.50):
        self.threshold = threshold
        self.weights = np.array([0.26, 0.20, 0.18, 0.25, 0.11])

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        return self.predict_curve(X)[:, -1]

    def predict_curve(self, X):
        temp_rate = np.diff(X[:, :, 0], axis=1, prepend=X[:, :1, 0])
        rate_norm = np.clip(temp_rate * 4, 0, 1)
        X2 = X.copy()
        X2[:, :, 0] = 0.75 * X[:, :, 0] + 0.25 * rate_norm
        weighted = np.tensordot(X2, self.weights, axes=([2], [0]))
        strongest_sensor = X2[:, :, :4].max(axis=2)
        risk = 0.65 * weighted + 0.35 * strongest_sensor
        return np.clip(risk, 0, 1)

    def predict(self, X):
        return (self.predict_proba(X) >= self.threshold).astype(int)


class MLPClassifier:
    def __init__(self, hidden=20, lr=0.05, epochs=220, seed=2026):
        self.hidden = hidden
        self.lr = lr
        self.epochs = epochs
        self.seed = seed
        self.scaler = StandardScaler2D()

    def fit(self, X, y):
        F = self.scaler.fit_transform(extract_stat_features(X))
        y = y.reshape(-1, 1)
        pos = max(float(y.sum()), 1.0)
        neg = max(float(len(y) - y.sum()), 1.0)
        sample_weight = np.where(y == 1, len(y) / (2 * pos), len(y) / (2 * neg))
        rng = np.random.default_rng(self.seed)
        self.W1 = rng.normal(0, 0.18, (F.shape[1], self.hidden))
        self.b1 = np.zeros((1, self.hidden))
        self.W2 = rng.normal(0, 0.18, (self.hidden, 1))
        self.b2 = np.zeros((1, 1))

        n = len(F)
        for _ in range(self.epochs):
            h = np.tanh(F @ self.W1 + self.b1)
            p = sigmoid(h @ self.W2 + self.b2)
            dz2 = (p - y) * sample_weight / n
            dW2 = h.T @ dz2
            db2 = dz2.sum(axis=0, keepdims=True)
            dh = dz2 @ self.W2.T
            dz1 = dh * (1 - h ** 2)
            dW1 = F.T @ dz1
            db1 = dz1.sum(axis=0, keepdims=True)
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2
        return self

    def _proba_from_features(self, F):
        F = self.scaler.transform(F)
        h = np.tanh(F @ self.W1 + self.b1)
        return sigmoid(h @ self.W2 + self.b2).ravel()

    def predict_proba(self, X):
        return self._proba_from_features(extract_stat_features(X))

    def predict_curve(self, X):
        curves = []
        for t in range(5, X.shape[1] + 1):
            curves.append(self.predict_proba(X[:, :t, :]))
        prefix = np.zeros((X.shape[0], 4))
        return np.concatenate([prefix, np.vstack(curves).T], axis=1)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


class LinearSVMClassifier:
    def __init__(self, lr=0.01, epochs=260, c=1.0):
        self.lr = lr
        self.epochs = epochs
        self.c = c
        self.scaler = StandardScaler2D()

    def fit(self, X, y):
        F = self.scaler.fit_transform(extract_stat_features(X))
        y2 = np.where(y == 1, 1.0, -1.0)
        self.w = np.zeros(F.shape[1])
        self.b = 0.0
        n = len(F)
        for _ in range(self.epochs):
            margins = y2 * (F @ self.w + self.b)
            mask = margins < 1
            grad_w = self.w - self.c * (F[mask] * y2[mask, None]).sum(axis=0) / n
            grad_b = -self.c * y2[mask].sum() / n
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
        return self

    def decision_function(self, X):
        F = self.scaler.transform(extract_stat_features(X))
        return F @ self.w + self.b

    def predict_proba(self, X):
        return sigmoid(self.decision_function(X))

    def predict_curve(self, X):
        curves = []
        for t in range(5, X.shape[1] + 1):
            curves.append(self.predict_proba(X[:, :t, :]))
        prefix = np.zeros((X.shape[0], 4))
        return np.concatenate([prefix, np.vstack(curves).T], axis=1)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


class NumpyLSTMClassifier:
    def __init__(self, hidden=14, lr=0.06, epochs=90, seed=2026):
        self.hidden = hidden
        self.lr = lr
        self.epochs = epochs
        self.seed = seed

    def _init_params(self, input_dim):
        rng = np.random.default_rng(self.seed)
        scale = 1.0 / np.sqrt(input_dim + self.hidden)
        zdim = input_dim + self.hidden
        self.Wf = rng.normal(0, scale, (zdim, self.hidden))
        self.Wi = rng.normal(0, scale, (zdim, self.hidden))
        self.Wc = rng.normal(0, scale, (zdim, self.hidden))
        self.Wo = rng.normal(0, scale, (zdim, self.hidden))
        self.bf = np.ones((1, self.hidden)) * 0.5
        self.bi = np.zeros((1, self.hidden))
        self.bc = np.zeros((1, self.hidden))
        self.bo = np.zeros((1, self.hidden))
        self.Wy = rng.normal(0, scale, (self.hidden, 1))
        self.by = np.zeros((1, 1))

    def _forward(self, X, keep_cache=False):
        n, steps, _ = X.shape
        h = np.zeros((n, self.hidden))
        c = np.zeros((n, self.hidden))
        caches = []
        for t in range(steps):
            z = np.concatenate([h, X[:, t, :]], axis=1)
            f = sigmoid(z @ self.Wf + self.bf)
            i = sigmoid(z @ self.Wi + self.bi)
            g = np.tanh(z @ self.Wc + self.bc)
            c_prev = c
            c = f * c + i * g
            o = sigmoid(z @ self.Wo + self.bo)
            h_prev = h
            h = o * np.tanh(c)
            if keep_cache:
                caches.append((z, f, i, g, o, c, c_prev, h_prev))
        p = sigmoid(h @ self.Wy + self.by)
        return (p, h, caches) if keep_cache else p

    def fit(self, X, y):
        self._init_params(X.shape[-1])
        y = y.reshape(-1, 1)
        pos = max(float(y.sum()), 1.0)
        neg = max(float(len(y) - y.sum()), 1.0)
        sample_weight = np.where(y == 1, len(y) / (2 * pos), len(y) / (2 * neg))
        n = len(X)
        for _ in range(self.epochs):
            p, h_last, caches = self._forward(X, keep_cache=True)
            dy = (p - y) * sample_weight / n
            grads = {name: np.zeros_like(getattr(self, name)) for name in [
                "Wf", "Wi", "Wc", "Wo", "Wy", "bf", "bi", "bc", "bo", "by"
            ]}
            grads["Wy"] = h_last.T @ dy
            grads["by"] = dy.sum(axis=0, keepdims=True)
            dh_next = dy @ self.Wy.T
            dc_next = np.zeros((n, self.hidden))
            for z, f, i, g, o, c, c_prev, _h_prev in reversed(caches):
                tanh_c = np.tanh(c)
                dh = dh_next
                do = dh * tanh_c
                do_raw = do * o * (1 - o)
                dc = dh * o * (1 - tanh_c ** 2) + dc_next
                df_raw = (dc * c_prev) * f * (1 - f)
                di_raw = (dc * g) * i * (1 - i)
                dg_raw = (dc * i) * (1 - g ** 2)
                grads["Wf"] += z.T @ df_raw
                grads["Wi"] += z.T @ di_raw
                grads["Wc"] += z.T @ dg_raw
                grads["Wo"] += z.T @ do_raw
                grads["bf"] += df_raw.sum(axis=0, keepdims=True)
                grads["bi"] += di_raw.sum(axis=0, keepdims=True)
                grads["bc"] += dg_raw.sum(axis=0, keepdims=True)
                grads["bo"] += do_raw.sum(axis=0, keepdims=True)
                dz = df_raw @ self.Wf.T + di_raw @ self.Wi.T + dg_raw @ self.Wc.T + do_raw @ self.Wo.T
                dh_next = dz[:, :self.hidden]
                dc_next = dc * f
            for name, grad in grads.items():
                grad = np.clip(grad, -3.0, 3.0)
                setattr(self, name, getattr(self, name) - self.lr * grad)
        return self

    def predict_proba(self, X):
        return self._forward(X).ravel()

    def predict_curve(self, X):
        curves = []
        for t in range(5, X.shape[1] + 1):
            curves.append(self.predict_proba(X[:, :t, :]))
        prefix = np.zeros((X.shape[0], 4))
        return np.concatenate([prefix, np.vstack(curves).T], axis=1)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)
