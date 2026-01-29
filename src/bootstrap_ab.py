import numpy as np
from sklearn.metrics import f1_score
from .config import BOOTSTRAP_ITER, RANDOM_SEED

def bootstrap_f1(y_true, y_predA, y_predB):
    np.random.seed(RANDOM_SEED)
    n = len(y_true)
    deltas = []

    for _ in range(BOOTSTRAP_ITER):
        idx = np.random.choice(n, n, replace=True)
        f1A = f1_score(y_true[idx], y_predA[idx], pos_label="ANOMALO")
        f1B = f1_score(y_true[idx], y_predB[idx], pos_label="ANOMALO")
        deltas.append(f1A - f1B)

    deltas = np.array(deltas)
    return {
        "delta_f1_mean": deltas.mean(),
        "ci_lower": np.percentile(deltas, 2.5),
        "ci_upper": np.percentile(deltas, 97.5),
        "p_value": (deltas <= 0).mean()
    }
