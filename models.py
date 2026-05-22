"""
models.py – Three ML models trained on processed_alerts.csv
Models: Random Forest · Logistic Regression · Gradient Boosting
"""
import pandas as pd
import numpy as np
import io, base64, warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing    import LabelEncoder, StandardScaler
from sklearn.metrics          import (accuracy_score, f1_score,
                                      confusion_matrix, classification_report)
from sklearn.ensemble         import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model     import LogisticRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── helpers ───────────────────────────────────────────────────────────────────
def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img

def _load_data():
    df = pd.read_csv('processed_alerts.csv')
    df = df.dropna(axis=1, how='all')
    for col in df.select_dtypes(include='object').columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    df = df.fillna(df.median(numeric_only=True))
    target = 'nivel_alerta' if 'nivel_alerta' in df.columns else df.columns[-1]
    X = df.drop(columns=[target])
    y = df[target]
    return X, y, target

def _split(X, y, test_size=0.2, seed=42):
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

def _confusion_img(y_test, y_pred, title):
    cm  = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_title(title)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    plt.colorbar(im, ax=ax)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i,j]), ha='center', va='center', color='black')
    return _fig_to_b64(fig)

def _importance_img(importances, feature_names, title):
    idx = np.argsort(importances)[-10:]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(range(len(idx)), importances[idx], color='steelblue')
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([feature_names[i] for i in idx])
    ax.set_title(title)
    ax.set_xlabel('Importance')
    return _fig_to_b64(fig)

def _cv_img(scores, model_name):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(range(1, len(scores)+1), scores, marker='o', color='steelblue')
    ax.axhline(scores.mean(), color='red', linestyle='--', label=f'Mean={scores.mean():.3f}')
    ax.set_title(f'{model_name} – Cross-Validation F1')
    ax.set_xlabel('Fold'); ax.set_ylabel('F1 Score')
    ax.legend()
    return _fig_to_b64(fig)

# ── Model 1: Random Forest ────────────────────────────────────────────────────
def run_random_forest():
    X, y, target = _load_data()
    X_train, X_test, y_train, y_test = _split(X, y)

    hyperparams = dict(n_estimators=200, max_depth=10, min_samples_split=5,
                       class_weight='balanced', random_state=42, n_jobs=-1)
    clf = RandomForestClassifier(**hyperparams)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    cv_scores = cross_val_score(clf, X, y, cv=5, scoring='f1_weighted')
    accuracy  = accuracy_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred, average='weighted')

    sample = X_test.iloc[:5].copy()
    sample['Actual']    = y_test.iloc[:5].values
    sample['Predicted'] = y_pred[:5]
    predictions_html = sample[['Actual','Predicted']].to_html(
        classes='table table-sm', border=0, index=False)

    confusion_b64  = _confusion_img(y_test, y_pred, 'Random Forest – Confusion Matrix')
    importance_b64 = _importance_img(clf.feature_importances_, X.columns.tolist(),
                                     'Random Forest – Feature Importance')
    cv_b64         = _cv_img(cv_scores, 'Random Forest')

    return dict(
        name='Random Forest', slug='random-forest', icon='🌲',
        show_metrics=True,
        description=(
            'Random Forest is an ensemble of decision trees trained on random subsets '
            'of features and data (bagging). Each tree votes independently and the '
            'majority class wins, which reduces overfitting and improves generalization.'
        ),
        explanation=[
            'Data is split 80/20 (train/test) with stratification to preserve class balance.',
            'Each tree uses a random subset of features at each split (sqrt rule).',
            'Hyperparameter tuning: n_estimators=200, max_depth=10, class_weight=balanced.',
            '5-fold cross-validation is used to validate the stability of the model.',
            'Feature importances are extracted to explain which variables drive predictions.',
        ],
        hyperparams=hyperparams,
        accuracy=round(accuracy, 4), f1=round(f1, 4),
        cv_mean=round(cv_scores.mean(), 4), cv_std=round(cv_scores.std(), 4),
        predictions_html=predictions_html,
        confusion_b64=confusion_b64, importance_b64=importance_b64, cv_b64=cv_b64,
        train_size=len(X_train), test_size=len(X_test),
    )

# ── Model 2: Logistic Regression (SIN métricas — próxima entrega) ─────────────
def run_logistic_regression():
    X, y, target = _load_data()
    X_train, X_test, y_train, y_test = _split(X, y)

    # Normalización de features
    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # Configuración de hiperparámetros
    hyperparams = dict(
        C=1.0,
        max_iter=500,
        solver='lbfgs',
        class_weight='balanced',
        multi_class='auto',
        random_state=42
    )
    clf = LogisticRegression(**hyperparams)

    # Entrenamiento
    clf.fit(X_train_s, y_train)

    # Predicciones sobre el conjunto de prueba
    y_pred = clf.predict(X_test_s)

    # Proceso de validación: cross-validation sobre el dataset completo
    cv_scores = cross_val_score(
        LogisticRegression(**hyperparams),
        scaler.fit_transform(X), y,
        cv=5, scoring='f1_weighted'
    )

    # Ejemplos de predicción (primeras 5 filas del test set)
    sample = X_test.iloc[:5].copy()
    sample['Actual']    = y_test.iloc[:5].values
    sample['Predicted'] = y_pred[:5]
    predictions_html = sample[['Actual', 'Predicted']].to_html(
        classes='table table-sm', border=0, index=False
    )

    # Visualización: magnitud de coeficientes (importancia de features)
    if clf.coef_.ndim > 1:
        importances = np.abs(clf.coef_).mean(axis=0)
    else:
        importances = np.abs(clf.coef_[0])
    importance_b64 = _importance_img(
        importances, X.columns.tolist(),
        'Logistic Regression – Coefficient Magnitude (Top 10 Features)'
    )

    # Visualización: distribución de clases predichas
    fig, ax = plt.subplots(figsize=(6, 4))
    unique, counts = np.unique(y_pred, return_counts=True)
    ax.bar([str(u) for u in unique], counts, color='#2d6a4f', edgecolor='white')
    ax.set_title('Logistic Regression – Predicted Class Distribution')
    ax.set_xlabel('Class'); ax.set_ylabel('Count')
    dist_b64 = _fig_to_b64(fig)

    # Visualización: curva de validación cruzada
    cv_b64 = _cv_img(cv_scores, 'Logistic Regression')

    return dict(
        name='Logistic Regression', slug='logistic-regression', icon='📈',
        show_metrics=False,   # métricas se agregan en la próxima entrega
        description=(
            'Logistic Regression modela la probabilidad de pertenencia a una clase '
            'usando la función sigmoide. Es un modelo lineal e interpretable que sirve '
            'como referencia base y funciona bien cuando los features están normalizados '
            'y las clases son razonablemente separables en el espacio de features.'
        ),
        explanation=[
            'Se cargan y limpian los datos desde processed_alerts.csv, codificando '
            'variables categóricas con LabelEncoder e imputando valores faltantes con la mediana.',
            'División del dataset: 80% entrenamiento / 20% prueba usando train_test_split '
            'con estratificación (stratify=y) para preservar la proporción de clases.',
            'Preprocesamiento: se aplica StandardScaler sobre el conjunto de entrenamiento '
            '(fit_transform) y sobre el de prueba (solo transform) para evitar data leakage.',
            'Configuración de hiperparámetros: C=1.0 controla la regularización L2, '
            'solver=lbfgs soporta multiclase nativamente, max_iter=500 asegura convergencia, '
            'class_weight=balanced corrige el desbalance de clases automáticamente.',
            'Entrenamiento: clf.fit(X_train_s, y_train) ajusta los coeficientes del modelo '
            'minimizando la función de pérdida log-loss sobre el conjunto de entrenamiento.',
            'Proceso de validación: cross_val_score con cv=5 divide el dataset en 5 folds, '
            'entrena en 4 y valida en 1 de forma rotativa, reportando F1 ponderado por fold.',
            'Generación de predicciones: clf.predict(X_test_s) produce la clase con mayor '
            'probabilidad para cada muestra del conjunto de prueba.',
            'Se extraen los coeficientes del modelo (clf.coef_) para identificar cuáles '
            'features tienen mayor influencia sobre las predicciones (interpretabilidad).',
        ],
        hyperparams=hyperparams,
        cv_mean=round(cv_scores.mean(), 4),
        cv_std=round(cv_scores.std(), 4),
        predictions_html=predictions_html,
        importance_b64=importance_b64,
        dist_b64=dist_b64,
        cv_b64=cv_b64,
        train_size=len(X_train),
        test_size=len(X_test),
    )

# ── Model 3: Gradient Boosting ────────────────────────────────────────────────
def run_gradient_boosting():
    X, y, target = _load_data()
    X_train, X_test, y_train, y_test = _split(X, y)

    hyperparams = dict(n_estimators=150, learning_rate=0.1, max_depth=5,
                       subsample=0.8, random_state=42)
    clf = GradientBoostingClassifier(**hyperparams)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    cv_scores = cross_val_score(clf, X, y, cv=5, scoring='f1_weighted')
    accuracy  = accuracy_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred, average='weighted')

    sample = X_test.iloc[:5].copy()
    sample['Actual']    = y_test.iloc[:5].values
    sample['Predicted'] = y_pred[:5]
    predictions_html = sample[['Actual','Predicted']].to_html(
        classes='table table-sm', border=0, index=False)

    confusion_b64  = _confusion_img(y_test, y_pred, 'Gradient Boosting – Confusion Matrix')
    importance_b64 = _importance_img(clf.feature_importances_, X.columns.tolist(),
                                     'Gradient Boosting – Feature Importance')
    cv_b64         = _cv_img(cv_scores, 'Gradient Boosting')

    return dict(
        name='Gradient Boosting', slug='gradient-boosting', icon='🚀',
        show_metrics=True,
        description=(
            'Gradient Boosting builds trees sequentially, where each new tree corrects '
            'the residual errors of the previous ones. It typically achieves higher '
            'accuracy than bagging methods at the cost of longer training time.'
        ),
        explanation=[
            'Trees are built sequentially; each one minimises the loss of the ensemble.',
            'learning_rate=0.1 shrinks each tree contribution to prevent overfitting.',
            'subsample=0.8 introduces stochastic sampling similar to Random Forest.',
            'max_depth=5 limits tree complexity to reduce variance.',
            '5-fold cross-validation validates robustness across different data splits.',
        ],
        hyperparams=hyperparams,
        accuracy=round(accuracy, 4), f1=round(f1, 4),
        cv_mean=round(cv_scores.mean(), 4), cv_std=round(cv_scores.std(), 4),
        predictions_html=predictions_html,
        confusion_b64=confusion_b64, importance_b64=importance_b64, cv_b64=cv_b64,
        train_size=len(X_train), test_size=len(X_test),
    )

# ── Run all (for Engineering summary page) ───────────────────────────────────
def run_all_models():
    return [
        run_random_forest(),
        run_logistic_regression(),
        run_gradient_boosting(),
    ]
