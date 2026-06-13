from dataclasses import dataclass
import numpy as np

from config import CLASS_NAMES, CRITICAL_TIME, N_PER_CLASS, SEED, SEQ_LEN


@dataclass
class Dataset:
    X: np.ndarray
    y_multi: np.ndarray
    y_binary: np.ndarray
    critical_time: np.ndarray


def _smooth_noise(rng, length, scale):
    noise = rng.normal(0.0, scale, length)
    kernel = np.ones(5) / 5
    return np.convolve(noise, kernel, mode="same")


def _ramp(t, start, end, power=1.3):
    z = np.clip((t - start) / max(end - start, 1), 0, 1)
    return z ** power


def _generate_one(rng, class_id, seq_len):
    t = np.arange(seq_len)
    ambient = 24 + rng.normal(0, 1.2)
    base_temp = ambient + _smooth_noise(rng, seq_len, 0.7)
    smoke = 0.03 + np.abs(_smooth_noise(rng, seq_len, 0.015))
    gas = 0.04 + np.abs(_smooth_noise(rng, seq_len, 0.018))
    ir = base_temp + rng.normal(0, 0.8, seq_len)
    current = 0.08 + np.abs(_smooth_noise(rng, seq_len, 0.025))

    if class_id == 0:
        temp = base_temp + 1.5 * _ramp(t, 42, 60)
        smoke += 0.02 * _ramp(t, 45, 60)
        gas += 0.02 * _ramp(t, 45, 60)
        ir = temp + rng.normal(0, 0.7, seq_len)
        critical = -1
    elif class_id == 1:
        temp = base_temp + 22 * _ramp(t, 18, 58) + rng.normal(0, 0.7, seq_len)
        smoke += 0.10 * _ramp(t, 36, 58)
        gas += 0.09 * _ramp(t, 34, 58)
        ir = temp + 7.0 * _ramp(t, 25, 56)
        current += 0.18 * _ramp(t, 20, 50)
        critical = CRITICAL_TIME
    elif class_id == 2:
        temp = base_temp + 12 * _ramp(t, 24, 60) + rng.normal(0, 0.6, seq_len)
        smoke += 0.42 * _ramp(t, 18, 56)
        gas += 0.36 * _ramp(t, 16, 55)
        ir = temp + 4.0 * _ramp(t, 30, 58)
        current += 0.10 * _ramp(t, 25, 55)
        critical = CRITICAL_TIME
    else:
        temp = base_temp + 42 * _ramp(t, 10, 50) + rng.normal(0, 1.1, seq_len)
        smoke += 0.70 * _ramp(t, 12, 47)
        gas += 0.55 * _ramp(t, 12, 46)
        ir = temp + 18.0 * _ramp(t, 14, 45)
        current += 0.28 * _ramp(t, 10, 42)
        critical = CRITICAL_TIME

    X = np.stack([temp, smoke, gas, ir, current], axis=1)
    X += rng.normal(0, [0.25, 0.006, 0.006, 0.35, 0.006], X.shape)
    return X, critical


def generate_dataset(n_per_class=N_PER_CLASS, seq_len=SEQ_LEN, seed=SEED):
    rng = np.random.default_rng(seed)
    xs, ys, critical = [], [], []
    for class_id in CLASS_NAMES:
        for _ in range(n_per_class):
            x, c = _generate_one(rng, class_id, seq_len)
            xs.append(x)
            ys.append(class_id)
            critical.append(c)

    X = np.asarray(xs, dtype=np.float64)
    y_multi = np.asarray(ys, dtype=np.int64)
    y_binary = (y_multi > 0).astype(np.int64)
    critical_time = np.asarray(critical, dtype=np.int64)

    order = rng.permutation(len(X))
    return Dataset(X[order], y_multi[order], y_binary[order], critical_time[order])


def split_dataset(dataset, train_ratio=0.7, val_ratio=0.1):
    n = len(dataset.X)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    idx_train = slice(0, n_train)
    idx_val = slice(n_train, n_train + n_val)
    idx_test = slice(n_train + n_val, n)

    def part(idx):
        return Dataset(
            dataset.X[idx],
            dataset.y_multi[idx],
            dataset.y_binary[idx],
            dataset.critical_time[idx],
        )

    return part(idx_train), part(idx_val), part(idx_test)

