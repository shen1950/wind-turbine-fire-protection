import csv
from pathlib import Path
import numpy as np

from config import ALARM_THRESHOLD


def classification_metrics(y_true, y_pred):
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    accuracy = (tp + tn) / max(len(y_true), 1)
    false_alarm = fp / max(fp + tn, 1)
    miss = fn / max(fn + tp, 1)
    return {
        "accuracy": accuracy,
        "false_alarm_rate": false_alarm,
        "miss_rate": miss,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def lead_time_seconds(risk_curve, y_true, critical_time, threshold=ALARM_THRESHOLD):
    leads = []
    response_times = []
    for curve, y, ct in zip(risk_curve, y_true, critical_time):
        if y == 0 or ct < 0:
            continue
        alarm_points = np.where(curve >= threshold)[0]
        if len(alarm_points) == 0:
            continue
        first_alarm = int(alarm_points[0])
        leads.append(max(0, int(ct) - first_alarm))
        response_times.append(first_alarm)
    if not leads:
        return 0.0, 0.0
    return float(np.mean(leads)), float(np.mean(response_times))


def evaluate_model(name, model, X, y, critical_time):
    proba = model.predict_proba(X)
    pred = (proba >= ALARM_THRESHOLD).astype(int)
    metrics = classification_metrics(y, pred)
    curve = model.predict_curve(X)
    lead, response = lead_time_seconds(curve, y, critical_time)
    metrics.update({
        "model": name,
        "avg_lead_time_s": lead,
        "avg_response_time_s": response,
    })
    return metrics


def save_metrics_csv(metrics, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model", "accuracy", "false_alarm_rate", "miss_rate",
        "avg_lead_time_s", "avg_response_time_s", "tp", "tn", "fp", "fn",
    ]
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in metrics:
            writer.writerow({k: row[k] for k in fields})

