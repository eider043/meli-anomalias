from sklearn.metrics import f1_score, precision_score, recall_score

def compute_metrics(y_true, y_pred):
    return {
        "f1": f1_score(y_true, y_pred, pos_label="ANOMALO"),
        "precision": precision_score(y_true, y_pred, pos_label="ANOMALO"),
        "recall": recall_score(y_true, y_pred, pos_label="ANOMALO")
    }
