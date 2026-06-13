from pathlib import Path

import numpy as np

from config import SEED, TRAIN_RATIO, VAL_RATIO
from data_generator import generate_dataset, split_dataset
from evaluate import evaluate_model, save_metrics_csv
from fuzzy_system import FuzzyFusionModel
from models import LinearSVMClassifier, MLPClassifier, NumpyLSTMClassifier, ThresholdRiskModel
from plot_results import save_metric_chart
from preprocessing import MinMaxScaler3D, smooth_sequence


OUT_DIR = Path("outputs")


def save_npz(train, val, test):
    OUT_DIR.mkdir(exist_ok=True)
    np.savez_compressed(
        OUT_DIR / "simulated_wind_turbine_fire_dataset.npz",
        X_train=train.X, y_train=train.y_binary, y_train_multi=train.y_multi,
        X_val=val.X, y_val=val.y_binary, y_val_multi=val.y_multi,
        X_test=test.X, y_test=test.y_binary, y_test_multi=test.y_multi,
        critical_time_test=test.critical_time,
    )


def main():
    np.random.seed(SEED)
    dataset = generate_dataset(seed=SEED)
    train, val, test = split_dataset(dataset, TRAIN_RATIO, VAL_RATIO)

    scaler = MinMaxScaler3D()
    X_train = scaler.fit_transform(smooth_sequence(train.X))
    X_val = scaler.transform(smooth_sequence(val.X))
    X_test = scaler.transform(smooth_sequence(test.X))
    train.X, val.X, test.X = X_train, X_val, X_test
    save_npz(train, val, test)

    models = []
    threshold = ThresholdRiskModel().fit(X_train, train.y_binary)
    models.append(("阈值法", threshold))

    bp = MLPClassifier(hidden=22, lr=0.04, epochs=240, seed=SEED).fit(X_train, train.y_binary)
    models.append(("BP网络", bp))

    svm = LinearSVMClassifier(lr=0.012, epochs=280, c=1.2).fit(X_train, train.y_binary)
    models.append(("SVM", svm))

    lstm = NumpyLSTMClassifier(hidden=14, lr=0.055, epochs=95, seed=SEED).fit(X_train, train.y_binary)
    models.append(("LSTM", lstm))

    fuzzy = FuzzyFusionModel(lstm).fit(X_train, train.y_binary)
    models.append(("LSTM+模糊", fuzzy))

    metrics = []
    print("模型评估结果：")
    for name, model in models:
        row = evaluate_model(name, model, X_test, test.y_binary, test.critical_time)
        metrics.append(row)
        print(
            f"{name:10s} 准确率={row['accuracy']*100:5.1f}% "
            f"误报率={row['false_alarm_rate']*100:4.1f}% "
            f"漏报率={row['miss_rate']*100:4.1f}% "
            f"平均提前量={row['avg_lead_time_s']:4.1f}s"
        )

    save_metrics_csv(metrics, OUT_DIR / "metrics.csv")
    save_metric_chart(metrics, OUT_DIR / "algorithm_comparison.png")
    print(f"\n输出目录：{OUT_DIR.resolve()}")
    print("已生成：metrics.csv、algorithm_comparison.png、simulated_wind_turbine_fire_dataset.npz")


if __name__ == "__main__":
    main()

