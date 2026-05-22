import pandas as pd
import numpy as np
import io, base64, warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img


def _pick_target(df):
    candidates = ['nivel_alerta', 'target', 'class', 'label', 'alert', 'pest', 'y', 'outcome', 'result', 'category_rain']
    for c in candidates:
        if c in df.columns and df[c].nunique(dropna=True) > 1:
            return c
    for c in reversed(df.columns.tolist()):
        if df[c].nunique(dropna=True) > 1:
            return c
    raise ValueError(f"No se encontró una columna objetivo con al menos 2 clases. Columnas: {list(df.columns)}")


def _load_data():
    df = pd.read_csv('processed_alerts.csv')
    df = df.dropna(axis=1, how='all')

    print("COLUMNAS DISPONIBLES:", list(df.columns))
    target = _pick_target(df)
    print("TARGET USADO:", target)
    print(df[target].value_counts(dropna=False))

    y_raw = df[target].copy()
    X = df.drop(columns=[target]).copy()

    if y_raw.dtype == 'object':
        y = LabelEncoder().fit_transform(y_raw.astype(str))
    else:
        y = pd.Series(y_raw).astype(int).values

    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = X[col].fillna('missing')
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        else:
            X[col] = X[col].fillna(X[col].median())

    if len(np.unique(y)) < 2:
        raise ValueError(f'El target "{target}" solo tiene una clase.')

    return X, pd.Series(y), target


def _split(X, y, test_size=0.2, seed=42):
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def _confusion_img(y_test, y_pred, title):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.colorbar(im, ax=ax)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='black')
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
    ax.plot(range(1, len(scores) + 1), scores, marker='o', color='steelblue')
    ax.axhline(scores.mean(), color='red', linestyle='--', label=f'Mean={scores.mean():.3f}')
    ax.set_title(f'{model_name} – Cross-Validation F1')
    ax.set_xlabel('Fold')
    ax.set_ylabel('F1 Score')
    ax.legend()
    return _fig_to_b64(fig)


def run_logistic_regression():
    try:
        X, y, target = _load_data()
        X_train, X_test, y_train, y_test = _split(X, y)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        hyperparams = dict(
            C=1.0,
            max_iter=500,
            solver='lbfgs',
            class_weight='balanced',
            random_state=42
        )

        clf = LogisticRegression(**hyperparams)
        clf.fit(X_train_s, y_train)
        y_pred = clf.predict(X_test_s)

        cv_scores = cross_val_score(
            LogisticRegression(**hyperparams),
            scaler.fit_transform(X),
            y,
            cv=5,
            scoring='f1_weighted'
        )

        sample = X_test.iloc[:5].copy()
        sample['Actual'] = y_test.iloc[:5].values
        sample['Predicted'] = y_pred[:5]
        predictions_html = sample[['Actual', 'Predicted']].to_html(
            classes='table table-sm', border=0, index=False
        )

        if clf.coef_.ndim > 1:
            importances = np.abs(clf.coef_).mean(axis=0)
        else:
            importances = np.abs(clf.coef_[0])

        importance_b64 = _importance_img(importances, X.columns.tolist(), 'Logistic Regression – Coefficient Magnitude')

        fig, ax = plt.subplots(figsize=(6, 4))
        unique, counts = np.unique(y_pred, return_counts=True)
        ax.bar([str(u) for u in unique], counts, color='#2d6a4f', edgecolor='white')
        ax.set_title('Logistic Regression – Predicted Class Distribution')
        ax.set_xlabel('Class')
        ax.set_ylabel('Count')
        dist_b64 = _fig_to_b64(fig)

        cv_b64 = _cv_img(cv_scores, 'Logistic Regression')

        return dict(
            name='Logistic Regression',
            slug='logistic-regression',
            icon='📈',
            show_metrics=False,
            description='Logistic Regression modela la probabilidad de pertenencia a una clase usando la función sigmoide.',
            explanation=[
                'Se cargan y limpian los datos desde processed_alerts.csv.',
                'División del dataset: 80% entrenamiento / 20% prueba con estratificación.',
                'Preprocesamiento: se aplica StandardScaler.',
                'Configuración: C=1.0, solver=lbfgs, max_iter=500, class_weight=balanced, random_state=42.',
                'Entrenamiento: clf.fit(X_train_s, y_train).',
                'Validación cruzada de 5 folds con cross_val_score.',
                'Predicción sobre el conjunto de prueba.',
                'Se extraen coeficientes para explicar el modelo.'
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
            error=None
        )
    except Exception as e:
        print("ERROR EN run_logistic_regression:", e)
        return dict(
            name='Logistic Regression',
            slug='logistic-regression',
            icon='📈',
            show_metrics=False,
            description='No se pudo generar el modelo.',
            explanation=[str(e)],
            hyperparams={},
            cv_mean=0,
            cv_std=0,
            predictions_html='',
            importance_b64='',
            dist_b64='',
            cv_b64='',
            train_size=0,
            test_size=0,
            error=str(e)
        )


def run_all_models():
    return [
        run_logistic_regression(),
    ]