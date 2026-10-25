"""Platform-owned scoring of saved predictions; no trial imports or model deserialization."""

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from sklearn import metrics  # type: ignore[import-untyped]


def score_predictions(
    y, predictions: pd.DataFrame, task_type: str, classes=(), groups=None
) -> dict:
    """Check shapes and finite values, then compute aggregate and optional per-group metrics.

    Undefined metrics are null with a reason, never made-up zeros. Probability columns follow the
    explicit class order; binary positive class is classes[1], matching sklearn convention.
    """
    y = np.asarray(y)
    if y.ndim != 1 or len(y) == 0 or len(predictions) != len(y):
        raise ValueError("prediction count must match a nonempty one-dimensional target")
    if "prediction" not in predictions or predictions["prediction"].isna().any():
        raise ValueError("missing predictions")
    pred = predictions["prediction"].to_numpy()
    unavailable = {}
    values: dict = {}
    if task_type == "regression":
        if set(predictions.columns) != {"prediction"}:
            raise ValueError("regression output must contain only the prediction column")
        if not np.isfinite(np.asarray(pred, dtype=float)).all():
            raise ValueError("predictions must be finite")
        values = {
            "mae": float(metrics.mean_absolute_error(y, pred)),
            "rmse": float(metrics.root_mean_squared_error(y, pred)),
            "r2": float(metrics.r2_score(y, pred)) if len(y) > 1 and len(set(y)) > 1 else None,
        }
        if len(y) < 2 or len(set(y)) < 2:
            unavailable["r2"] = "requires two observations and a nonconstant target"
    elif task_type in ("binary_classification", "multiclass_classification"):
        classes = list(classes)
        if len(classes) < 2 or len(set(classes)) != len(classes):
            raise ValueError("classification needs a unique, training-derived class order")
        if not set(pred).issubset(classes) or not set(y).issubset(classes):
            raise ValueError("labels outside the training class vocabulary")
        precision, recall, f1, support = metrics.precision_recall_fscore_support(
            y, pred, labels=classes, zero_division=0
        )
        values = {
            "accuracy": float(metrics.accuracy_score(y, pred)),
            "balanced_accuracy": float(np.mean(recall[np.asarray(support) > 0])),
            "precision_macro": float(np.mean(precision)),
            "recall_macro": float(np.mean(recall)),
            "f1_macro": float(np.mean(f1)),
        }
        if task_type == "binary_classification":
            values.update(precision=float(precision[1]), recall=float(recall[1]), f1=float(f1[1]))
        pcols = [f"p_{i}" for i in range(len(classes))]
        if set(predictions.columns) != {"prediction", *pcols}:
            raise ValueError("classification predictions require one probability column per class")
        probs = predictions[pcols].to_numpy(dtype=float)
        if not np.isfinite(probs).all() or (probs < 0).any() or (probs > 1).any():
            raise ValueError("probabilities must be finite and in [0, 1]")
        if not np.allclose(probs.sum(axis=1), 1, atol=1e-6, rtol=0):
            raise ValueError("probabilities must sum to one")
        values["log_loss"] = float(metrics.log_loss(y, probs, labels=classes))
        if len(set(y)) == len(classes):
            if len(classes) == 2:
                binary = (y == classes[1]).astype(int)
                values["roc_auc"] = float(metrics.roc_auc_score(binary, probs[:, 1]))
                values["pr_auc"] = float(metrics.average_precision_score(binary, probs[:, 1]))
                values["brier_score"] = float(metrics.brier_score_loss(binary, probs[:, 1]))
            else:
                values["roc_auc"] = float(
                    metrics.roc_auc_score(
                        y, probs, labels=classes, multi_class="ovr", average="macro"
                    )
                )
                truth = np.column_stack([y == label for label in classes])
                values["pr_auc"] = float(
                    metrics.average_precision_score(truth, probs, average="macro")
                )
        else:
            for key in ("roc_auc", "pr_auc"):
                values[key] = None
                unavailable[key] = "scoring partition does not contain every training class"
        values["confusion_matrix"] = metrics.confusion_matrix(y, pred, labels=classes).tolist()
        values["per_class"] = [
            {
                "label": label,
                "precision": float(p),
                "recall": float(r),
                "f1": float(f),
                "support": int(n),
            }
            for label, p, r, f, n in zip(classes, precision, recall, f1, support, strict=True)
        ]
    else:
        raise ValueError(f"unsupported tabular task: {task_type}")
    result = {"metrics": values, "unavailable": unavailable, "rows": len(y)}
    if groups is not None:
        groups = np.asarray(groups)
        if len(groups) != len(y):
            raise ValueError("group count must match target count")
        result["per_group"] = {
            str(group): score_predictions(
                y[groups == group], predictions.loc[groups == group], task_type, classes
            )
            for group in pd.unique(groups)
        }
    return result
