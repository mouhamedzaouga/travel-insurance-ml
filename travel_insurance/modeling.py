"""Prédiction du sinistre : modèles de classification sur données très déséquilibrées.

Toutes les étapes de préparation sont dans un Pipeline : elles sont donc
ré-apprises sur chaque pli d'entraînement (aucune fuite de données).
La métrique principale est la PR-AUC (average precision), plus informative
que l'accuracy quand seulement ~1 % des polices ont un sinistre.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from . import config as cfg
from .eda import _save

FEATURES = cfg.NUM_COLS + cfg.CAT_COLS


def _log1p_signed(x):
    return np.sign(x) * np.log1p(np.abs(x))


def build_preprocessor(scale: bool) -> ColumnTransformer:
    num_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True)),
                 ("log", FunctionTransformer(_log1p_signed))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), cfg.NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=30,
                              sparse_output=False), cfg.CAT_COLS),
    ])


def build_models() -> dict[str, Pipeline]:
    rs = cfg.RANDOM_STATE
    return {
        "Baseline (prior)": Pipeline([("pre", build_preprocessor(False)),
                                      ("clf", DummyClassifier(strategy="prior"))]),
        "Régression logistique": Pipeline([("pre", build_preprocessor(True)),
                                           ("clf", LogisticRegression(class_weight="balanced", C=0.3, max_iter=2000))]),
        "Random Forest": Pipeline([("pre", build_preprocessor(False)),
                                   ("clf", RandomForestClassifier(
                                       n_estimators=300, min_samples_leaf=10, max_features="sqrt",
                                       class_weight="balanced_subsample", n_jobs=-1, random_state=rs))]),
        "Gradient Boosting": Pipeline([("pre", build_preprocessor(False)),
                                       ("clf", HistGradientBoostingClassifier(
                                           max_depth=3, learning_rate=0.05, max_iter=200,
                                           l2_regularization=1.0, class_weight="balanced",
                                           random_state=rs))]),
    }


def lift_at(y, score, frac=0.1) -> float:
    """Part des sinistres capturés dans les `frac` polices les mieux classées."""
    k = int(len(y) * frac)
    top = np.argsort(-score)[:k]
    return float(np.asarray(y)[top].sum() / max(np.asarray(y).sum(), 1))


def run_modeling(df: pd.DataFrame, save: bool = True) -> dict:
    X, y = df[FEATURES], df[cfg.TARGET]
    cv = StratifiedKFold(5, shuffle=True, random_state=cfg.RANDOM_STATE)
    res: dict = {"cv": {}}

    for name, model in build_models().items():
        s = cross_validate(model, X, y, cv=cv, n_jobs=1,
                           scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision"})
        res["cv"][name] = {
            "roc_auc": f"{s['test_roc_auc'].mean():.3f} ± {s['test_roc_auc'].std():.3f}",
            "pr_auc": f"{s['test_pr_auc'].mean():.3f} ± {s['test_pr_auc'].std():.3f}",
            "_roc": float(s["test_roc_auc"].mean()), "_pr": float(s["test_pr_auc"].mean()),
        }
    res["baseline_pr_auc"] = round(float(y.mean()), 4)
    best = max((k for k in res["cv"] if k != "Baseline (prior)"), key=lambda k: res["cv"][k]["_pr"])
    res["best_model"] = best

    # --- évaluation hold-out du meilleur modèle -----------------------------
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y,
                                          random_state=cfg.RANDOM_STATE)
    model = build_models()[best].fit(Xtr, ytr)
    proba = model.predict_proba(Xte)[:, 1]
    res["holdout"] = {
        "n_test": int(len(yte)), "claims_test": int(yte.sum()),
        "roc_auc": round(float(roc_auc_score(yte, proba)), 3),
        "pr_auc": round(float(average_precision_score(yte, proba)), 3),
        "recall_in_top10pct": round(lift_at(yte, proba, 0.10), 3),
        "recall_in_top20pct": round(lift_at(yte, proba, 0.20), 3),
    }

    pi = permutation_importance(model, Xte, yte, scoring="average_precision",
                                n_repeats=20, random_state=cfg.RANDOM_STATE, n_jobs=1)
    imp = pd.Series(pi.importances_mean, index=FEATURES).sort_values()
    res["importance"] = imp.round(4).sort_values(ascending=False).to_dict()

    # --- figures ------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    names = [k for k in res["cv"] if k != "Baseline (prior)"]
    axes[0].barh(names + ["Baseline (prior)"],
                 [res["cv"][k]["_pr"] for k in names] + [res["cv"]["Baseline (prior)"]["_pr"]],
                 color=cfg.COLORS[0])
    axes[0].set_title("PR-AUC en validation croisée (5 plis)"); axes[0].set_xlabel("average precision")
    p, r, _ = precision_recall_curve(yte, proba)
    axes[1].plot(r, p, color=cfg.COLORS[1])
    axes[1].axhline(yte.mean(), color="grey", ls="--", label="hasard")
    axes[1].set_xlabel("Rappel"); axes[1].set_ylabel("Précision"); axes[1].legend()
    axes[1].set_title(f"Courbe précision-rappel – {best} (hold-out)")
    _save(fig, "11_model_comparison.png", save); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(imp.index, imp.values, color=cfg.COLORS[2])
    ax.set_title("Importance par permutation (baisse de PR-AUC)")
    _save(fig, "12_feature_importance.png", save); plt.close(fig)
    return res
