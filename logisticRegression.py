import pandas as pd
import numpy as np
import io
import base64
import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc
)
from sklearn.linear_model import LogisticRegression

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img


def _pick_target(df):
    candidates = [
        "nivel_alerta", "target", "class", "label", "alert",
        "pest", "y", "outcome", "result", "category_rain"
    ]

    for col in candidates:
        if col in df.columns and df[col].nunique(dropna=True) > 1:
            return col

    for col in reversed(df.columns.tolist()):
        if df[col].nunique(dropna=True) > 1:
            return col

    raise ValueError(
        f"No valid target column found with at least 2 classes. Columns: {list(df.columns)}"
    )


def _load_data():
    df = pd.read_csv("processed_alerts.csv")
    df = df.dropna(axis=1, how="all")

    print("AVAILABLE COLUMNS:", list(df.columns))

    target = _pick_target(df)
    print("SELECTED TARGET:", target)
    print("TARGET DISTRIBUTION:")
    print(df[target].value_counts(dropna=False))

    y_raw = df[target].copy()
    X = df.drop(columns=[target]).copy()

    if y_raw.dtype == "object":
        y = LabelEncoder().fit_transform(y_raw.astype(str))
    else:
        y = pd.Series(y_raw).astype(int).values

    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = X[col].fillna("missing")
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        else:
            X[col] = X[col].fillna(X[col].median())

    if len(np.unique(y)) < 2:
        raise ValueError(f'The target "{target}" contains only one class.')

    return X, pd.Series(y), target


def _split(X, y, test_size=0.2, seed=42):
    return train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )


def _confusion_img(y_test, y_pred, title):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.colorbar(im, ax=ax)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

    return _fig_to_b64(fig)


def _importance_img(importances, feature_names, title):
    idx = np.argsort(importances)[-10:]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(range(len(idx)), importances[idx], color="steelblue")
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([feature_names[i] for i in idx])
    ax.set_title(title)
    ax.set_xlabel("Importance")
    return _fig_to_b64(fig)


def _cv_img(scores, model_name):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(range(1, len(scores) + 1), scores, marker="o", color="steelblue")
    ax.axhline(scores.mean(), color="red", linestyle="--", label=f"Mean={scores.mean():.3f}")
    ax.set_title(f"{model_name} - Cross-Validation F1")
    ax.set_xlabel("Fold")
    ax.set_ylabel("F1 Score")
    ax.legend()
    return _fig_to_b64(fig)


def _roc_img(y_test, y_proba, n_classes, title):
    fig, ax = plt.subplots(figsize=(6, 4))

    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    else:
        classes = np.unique(y_test)
        y_bin = label_binarize(y_test, classes=classes)
        colors = plt.cm.Set1(np.linspace(0, 1, n_classes))

        for i, color in enumerate(colors):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=2, label=f"Class {classes[i]} (AUC={roc_auc:.3f})")

        ax.plot([0, 1], [0, 1], "k--", lw=1)

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    return _fig_to_b64(fig)


def _metrics_bar_img(metrics_dict, title):
    labels = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    colors = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0"]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.bar(labels, values, color=colors[:len(labels)], edgecolor="white", width=0.5)
    ax.set_ylim(0, 1.1)
    ax.set_title(title)
    ax.set_ylabel("Score")

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold"
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return _fig_to_b64(fig)


def run_logistic_regression():
    try:
        X, y, target = _load_data()
        X_train, X_test, y_train, y_test = _split(X, y)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        hyperparams = {
            "C": 1.0,
            "max_iter": 500,
            "solver": "lbfgs",
            "class_weight": "balanced",
            "random_state": 42
        }

        clf = LogisticRegression(**hyperparams)
        clf.fit(X_train_s, y_train)

        y_pred = clf.predict(X_test_s)
        y_proba = clf.predict_proba(X_test_s)

        n_classes = len(np.unique(y))
        avg = "binary" if n_classes == 2 else "weighted"

        accuracy = round(accuracy_score(y_test, y_pred), 4)
        precision = round(precision_score(y_test, y_pred, average=avg, zero_division=0), 4)
        recall = round(recall_score(y_test, y_pred, average=avg, zero_division=0), 4)
        f1 = round(f1_score(y_test, y_pred, average=avg, zero_division=0), 4)

        if n_classes == 2:
            fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
            roc_auc_val = round(auc(fpr, tpr), 4)
        else:
            classes = np.unique(y)
            y_bin = label_binarize(y_test, classes=classes)
            roc_aucs = []

            for i in range(n_classes):
                try:
                    fa, ta, _ = roc_curve(y_bin[:, i], y_proba[:, i])
                    roc_aucs.append(auc(fa, ta))
                except Exception:
                    pass

            roc_auc_val = round(np.mean(roc_aucs), 4) if roc_aucs else 0.0

        report_dict = classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0
        )
        report_df = pd.DataFrame(report_dict).transpose().round(4)
        classification_report_html = report_df.to_html(
            classes="table table-sm table-striped",
            border=0
        )

        cv_scores = cross_val_score(
            LogisticRegression(**hyperparams),
            scaler.fit_transform(X),
            y,
            cv=5,
            scoring="f1_weighted"
        )

        sample = X_test.iloc[:5].copy()
        sample["Actual"] = y_test.iloc[:5].values
        sample["Predicted"] = y_pred[:5]
        predictions_html = sample[["Actual", "Predicted"]].to_html(
            classes="table table-sm",
            border=0,
            index=False
        )

        if clf.coef_.ndim > 1:
            importances = np.abs(clf.coef_).mean(axis=0)
        else:
            importances = np.abs(clf.coef_[0])

        return {
            "name": "Logistic Regression",
            "slug": "logistic-regression",
            "icon": "📈",
            "show_metrics": True,
            "description": "Logistic Regression models the probability of class membership using the sigmoid function.",
            "explanation": [
                "Loads and cleans the dataset from processed_alerts.csv.",
                "Splits the dataset into 80% training and 20% testing using stratification.",
                "Applies feature scaling with StandardScaler.",
                "Fits the Logistic Regression model and evaluates classification performance."
            ],
            "hyperparams": hyperparams,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc_val,
            "classification_report_html": classification_report_html,
            "cv_mean": round(cv_scores.mean(), 4),
            "cv_std": round(cv_scores.std(), 4),
            "cv_scores": cv_scores.tolist(),
            "confusion_b64": _confusion_img(y_test, y_pred, "Logistic Regression - Confusion Matrix"),
            "roc_b64": _roc_img(y_test, y_proba, n_classes, "Logistic Regression - ROC Curve"),
            "metrics_bar_b64": _metrics_bar_img(
                {
                    "Accuracy": accuracy,
                    "Precision": precision,
                    "Recall": recall,
                    "F1": f1
                },
                "Logistic Regression - Classification Metrics"
            ),
            "importance_b64": _importance_img(
                importances,
                X.columns.tolist(),
                "Logistic Regression - Coefficient Magnitude"
            ),
            "cv_b64": _cv_img(cv_scores, "Logistic Regression"),
            "predictions_html": predictions_html,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "n_features": X.shape[1],
            "n_classes": n_classes,
            "target_name": target,
            "error": None
        }

    except Exception as e:
        print("ERROR IN run_logistic_regression():", str(e))
        return {
            "name": "Logistic Regression",
            "slug": "logistic-regression",
            "icon": "📈",
            "show_metrics": False,
            "description": "The model could not be generated.",
            "explanation": [str(e)],
            "hyperparams": {},
            "accuracy": 0,
            "precision": 0,
            "recall": 0,
            "f1_score": 0,
            "roc_auc": 0,
            "classification_report_html": "",
            "cv_mean": 0,
            "cv_std": 0,
            "cv_scores": [],
            "confusion_b64": "",
            "roc_b64": "",
            "metrics_bar_b64": "",
            "importance_b64": "",
            "cv_b64": "",
            "predictions_html": "",
            "train_size": 0,
            "test_size": 0,
            "n_features": 0,
            "n_classes": 0,
            "target_name": "",
            "error": str(e)
        }


def run_all_models():
    result = run_logistic_regression()
    print("RUN ALL MODELS RESULT ERROR:", result.get("error"))
    return [result]