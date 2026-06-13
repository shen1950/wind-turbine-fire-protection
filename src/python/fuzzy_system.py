import numpy as np


def _rising_membership(x, low, high):
    return np.clip((x - low) / max(high - low, 1e-8), 0.0, 1.0)


class FuzzyFusionModel:
    def __init__(self, lstm_model, threshold=0.50):
        self.lstm_model = lstm_model
        self.threshold = threshold
        self.weights = np.array([0.20, 0.18, 0.18, 0.14, 0.30])

    def fit(self, X, y):
        return self

    def _sensor_risk_curve(self, X):
        temp = X[:, :, 0]
        smoke = X[:, :, 1]
        gas = X[:, :, 2]
        ir = X[:, :, 3]
        temp_rate = np.diff(temp, axis=1, prepend=temp[:, :1])
        mu_t = _rising_membership(temp, 0.38, 0.72)
        mu_v = _rising_membership(temp_rate, 0.015, 0.065)
        mu_s = _rising_membership(smoke, 0.16, 0.55)
        mu_g = _rising_membership(gas, 0.16, 0.50)
        mu_ir = _rising_membership(ir, 0.42, 0.78)
        return np.stack([mu_t, mu_v, mu_s, mu_g, mu_ir], axis=2)

    def predict_curve(self, X):
        sensor_risk = self._sensor_risk_curve(X)
        lstm_curve = self.lstm_model.predict_curve(X)
        parts = np.zeros((X.shape[0], X.shape[1], 5))
        parts[:, :, :4] = sensor_risk[:, :, :4]
        parts[:, :, 4] = lstm_curve
        risk = np.tensordot(parts, self.weights, axes=([2], [0]))
        hot_bonus = 0.12 * sensor_risk[:, :, 4]
        consistency_bonus = 0.08 * ((sensor_risk[:, :, 0] > 0.6) & (sensor_risk[:, :, 2] > 0.5))
        engineering_guard = 0.92 * lstm_curve + 0.05 * sensor_risk.max(axis=2)
        return np.clip(np.maximum(risk + hot_bonus + consistency_bonus, engineering_guard), 0, 1)

    def predict_proba(self, X):
        return self.predict_curve(X)[:, -1]

    def predict(self, X):
        return (self.predict_proba(X) >= self.threshold).astype(int)
