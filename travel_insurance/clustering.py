"""Segmentation non supervisée : K-Means et classification ascendante hiérarchique."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from . import config as cfg
from .eda import _save

CLUSTER_FEATURES = ["Duration", "Net_Sales", "Commission", "Age", "Is_Airline"]


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Features de clustering : imputation médiane, winsorisation p1-p99, standardisation.

    La winsorisation n'est appliquée qu'ici (K-Means est sensible aux extrêmes) ;
    la variable cible n'est JAMAIS utilisée pour construire les clusters.
    """
    X = df[cfg.NUM_COLS].copy()
    X["Age"] = X["Age"].fillna(X["Age"].median())
    X = X.clip(X.quantile(0.01), X.quantile(0.99), axis=1)
    X["Is_Airline"] = (df["Agency_Type"] == "Airlines").astype(float)
    return X[CLUSTER_FEATURES], StandardScaler().fit_transform(X[CLUSTER_FEATURES])


def choose_k(Xs: np.ndarray, k_range=range(2, 9), sample=6000):
    rng = np.random.default_rng(cfg.RANDOM_STATE)
    idx = rng.choice(len(Xs), size=min(sample, len(Xs)), replace=False)
    inertia, sil = [], []
    for k in k_range:
        km = KMeans(k, n_init=10, random_state=cfg.RANDOM_STATE).fit(Xs)
        inertia.append(km.inertia_)
        sil.append(silhouette_score(Xs[idx], km.labels_[idx]))
    return list(k_range), inertia, sil


def run_clustering(df: pd.DataFrame, save: bool = True) -> tuple[pd.DataFrame, dict]:
    X, Xs = build_features(df)
    res: dict = {}

    ks, inertia, sil = choose_k(Xs)
    k_best = int(ks[int(np.argmax(sil))])
    res["k_candidates"] = {"k": ks, "silhouette": [round(s, 3) for s in sil]}
    res["k_best_silhouette"] = k_best
    K = 4                       # choix métier : 4 segments lisibles (voir README)
    res["k_used"] = K

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(ks, inertia, "o-", color=cfg.COLORS[0]); axes[0].set_title("Coude (inertie)")
    axes[1].plot(ks, sil, "s-", color=cfg.COLORS[1]); axes[1].set_title("Silhouette")
    for ax in axes:
        ax.axvline(K, color="red", ls="--", alpha=.6, label=f"k={K} retenu"); ax.set_xlabel("k"); ax.legend()
    _save(fig, "07_elbow_silhouette.png", save); plt.close(fig)

    # --- K-Means ----------------------------------------------------------
    km = KMeans(K, n_init=20, random_state=cfg.RANDOM_STATE).fit(Xs)
    rng = np.random.default_rng(cfg.RANDOM_STATE)
    idx = rng.choice(len(Xs), size=6000, replace=False)
    res["silhouette_kmeans"] = round(float(silhouette_score(Xs[idx], km.labels_[idx])), 3)

    # --- CAH (Ward) sur échantillon, étendue au reste par plus proche voisin ---
    samp = rng.choice(len(Xs), size=2500, replace=False)
    Z = linkage(Xs[samp], method="ward")
    lab_s = fcluster(Z, t=K, criterion="maxclust")
    knn = KNeighborsClassifier(n_neighbors=5).fit(Xs[samp], lab_s)
    cah = knn.predict(Xs)
    res["silhouette_cah"] = round(float(silhouette_score(Xs[idx], cah[idx])), 3)
    res["ari_kmeans_vs_cah"] = round(float(adjusted_rand_score(km.labels_, cah)), 3)

    fig, ax = plt.subplots(figsize=(13, 5))
    dendrogram(Z, ax=ax, truncate_mode="lastp", p=30, leaf_rotation=90, color_threshold=Z[-(K - 1), 2])
    ax.set_title("Dendrogramme (Ward, échantillon de 2 500 polices)")
    _save(fig, "08_dendrogram.png", save); plt.close(fig)

    # --- Profils : clusters triés par Net_Sales croissant pour des numéros stables
    out = df.copy()
    out["Cluster"] = km.labels_
    order = out.groupby("Cluster")["Net_Sales"].median().sort_values().index
    out["Cluster"] = out["Cluster"].map({old: new for new, old in enumerate(order)})
    out["Cluster_CAH"] = cah

    prof = out.groupby("Cluster").agg(
        n=("Claim", "size"),
        duration_med=("Duration", "median"), net_sales_med=("Net_Sales", "median"),
        commission_med=("Commission", "median"), age_med=("Age", "median"),
        airline_pct=("Agency_Type", lambda s: (s == "Airlines").mean() * 100),
        claims=("Claim", "sum"), claim_rate_pct=("Claim", lambda s: s.mean() * 100),
    ).round(2)
    res["profiles"] = prof.reset_index().to_dict("records")

    fig, ax = plt.subplots(figsize=(10, 4))
    cols = ["duration_med", "net_sales_med", "commission_med", "age_med", "airline_pct"]
    z = (prof[cols] - prof[cols].min()) / (prof[cols].max() - prof[cols].min() + 1e-9)
    sns.heatmap(z, annot=prof[cols], fmt=".1f", cmap="YlOrRd", cbar=False, ax=ax)
    ax.set_title("Profil des segments (médianes ; couleur = rang relatif)")
    _save(fig, "09_cluster_profiles.png", save); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(x=prof.index, y=prof["claim_rate_pct"], color=cfg.COLORS[0], ax=ax)
    for i, (r, c, n) in enumerate(zip(prof.claim_rate_pct, prof.claims, prof.n)):
        ax.text(i, r, f"{c:.0f}/{n:,.0f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(df.Claim.mean() * 100, color="red", ls="--", label="moyenne globale")
    ax.set_ylabel("Taux de sinistre (%)"); ax.set_xlabel("Segment"); ax.legend()
    ax.set_title("Taux de sinistre par segment (sinistres / polices)")
    _save(fig, "10_claim_rate_by_cluster.png", save); plt.close(fig)
    return out, res
