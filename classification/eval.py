from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import label_binarize


def _unwrap_estimator(model: Any) -> Any:
    if hasattr(model, "model"):
        return getattr(model, "model")
    return model


def evaluate_classifier(
    model: Any,
    x_test: np.ndarray,
    y_test: np.ndarray,
    class_names: Optional[Sequence[str]] = None,
    average: str = "macro",
    zero_division: int = 0,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Evaluate a trained classifier and return common metrics."""
    estimator = _unwrap_estimator(model)
    y_pred = estimator.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    balanced = balanced_accuracy_score(y_test, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test,
        y_pred,
        average=average,
        zero_division=zero_division,
    )
    report = classification_report(
        y_test,
        y_pred,
        target_names=list(class_names) if class_names is not None else None,
        zero_division=zero_division,
    )
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "support": support.tolist() if hasattr(support, "tolist") else support,
        "classification_report": report,
        "confusion_matrix": cm,
        "y_true": y_test,
        "y_pred": y_pred,
    }

    if verbose:
        print("Evaluation results")
        print("--------------")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Balanced accuracy: {balanced:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 score: {f1:.4f}")
        print("\nClassification report:\n", report)

    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[Sequence[str]] = None,
    normalize: bool = False,
    figsize: Tuple[int, int] = (8, 8),
    cmap: str = "Blues",
) -> plt.Figure:
    """Plot a confusion matrix for classifier predictions."""
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    labels = list(class_names) if class_names is not None else [str(i) for i in range(cm.shape[0])]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    return fig


def plot_training_loss(model: Any, figsize: Tuple[int, int] = (8, 5)) -> Optional[plt.Figure]:
    """Plot the training loss curve for sklearn MLPClassifier models."""
    estimator = _unwrap_estimator(model)
    loss_curve = getattr(estimator, "loss_curve_", None)
    if loss_curve is None:
        print("No loss curve available for this model.")
        return None

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(loss_curve, label="Training loss")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Curve")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_roc_auc(
    model: Any,
    x_test: np.ndarray,
    y_test: np.ndarray,
    class_names: Optional[Sequence[str]] = None,
    figsize: Tuple[int, int] = (8, 6),
) -> Optional[plt.Figure]:
    """Plot ROC AUC curve for binary or multiclass classifiers."""
    estimator = _unwrap_estimator(model)
    if not hasattr(estimator, "predict_proba"):
        print("ROC AUC plot requires a classifier with predict_proba().")
        return None

    y_score = estimator.predict_proba(x_test)
    labels = np.unique(y_test)
    if class_names is not None and len(class_names) == len(labels):
        labels = np.array(class_names)
    else:
        labels = labels.astype(str)

    y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
    n_classes = y_test_bin.shape[1]

    fig, ax = plt.subplots(figsize=figsize)
    if n_classes == 1:
        fpr, tpr, _ = roc_curve(y_test, y_score[:, 1])
        ax.plot(fpr, tpr, lw=2, label=f"ROC curve (area = {roc_auc_score(y_test, y_score[:, 1]):.3f})")
    else:
        for i in range(n_classes):
            ax.plot(
                *roc_curve(y_test_bin[:, i], y_score[:, i])[:2],
                lw=2,
                label=f"{labels[i]} (AUC = {roc_auc_score(y_test_bin[:, i], y_score[:, i]):.3f})",
            )
        macro_auc = roc_auc_score(y_test_bin, y_score, average="macro")
        ax.plot([], [], " ", label=f"Macro AUC = {macro_auc:.3f}")

    ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic")
    ax.legend(loc="lower right")
    ax.grid(True)
    fig.tight_layout()
    return fig


def plot_precision_recall(
    model: Any,
    x_test: np.ndarray,
    y_test: np.ndarray,
    class_names: Optional[Sequence[str]] = None,
    figsize: Tuple[int, int] = (8, 6),
) -> Optional[plt.Figure]:
    """Plot precision-recall curves for binary or multiclass classifiers."""
    estimator = _unwrap_estimator(model)
    if not hasattr(estimator, "predict_proba"):
        print("Precision-recall plot requires a classifier with predict_proba().")
        return None

    y_score = estimator.predict_proba(x_test)
    labels = np.unique(y_test)
    if class_names is not None and len(class_names) == len(labels):
        labels = np.array(class_names)
    else:
        labels = labels.astype(str)

    y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
    n_classes = y_test_bin.shape[1]

    fig, ax = plt.subplots(figsize=figsize)
    if n_classes == 1:
        precision, recall, _ = precision_recall_curve(y_test, y_score[:, 1])
        ax.plot(recall, precision, lw=2, label="Precision-Recall")
    else:
        for i in range(n_classes):
            precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
            ax.plot(recall, precision, lw=2, label=f"{labels[i]}")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="best")
    ax.grid(True)
    fig.tight_layout()
    return fig


def cross_validate_classifier(
    model: Any,
    x: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    scoring: str = "accuracy",
) -> Dict[str, Any]:
    """Run stratified cross-validation on a classifier."""
    estimator = _unwrap_estimator(model)
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(estimator, x, y, cv=splitter, scoring=scoring)

    return {
        "cv_scores": scores,
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
    }
