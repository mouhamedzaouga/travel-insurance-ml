"""Analyse exploratoire : statistiques, tests d'hypothèses et figures."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from . import config as cfg

sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.figsize": (10, 6), "font.size": 11})


def _save(fig, name, save):
    fig.tight_layout()
    if save:
        cfg.FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(cfg.FIG_DIR / name, dpi=130)
    return fig


def claim_rate_by(df: pd.DataFrame, col: str, min_n: int = 0) -> pd.DataFrame:
    g = df.groupby(col)["Claim"].agg(n="size", claims="sum", rate="mean")
    g["rate"] = (g["rate"] * 100).round(2)
    return g[g["n"] >= min_n].sort_values("rate", ascending=False)


def chi2_claim(df: pd.DataFrame, col: str) -> dict:
    """Test du khi-deux d'indépendance entre `col` et la survenue d'un sinistre."""
    table = pd.crosstab(df[col], df["Claim"])
    chi2, p, dof, expected = stats.chi2_contingency(table)
    return {"chi2": round(float(chi2), 2), "dof": int(dof), "p_value": float(p),
            "min_expected": round(float(expected.min()), 2)}


def mannwhitney_claim(df: pd.DataFrame, col: str) -> dict:
    """Mann-Whitney : la distribution de `col` diffère-t-elle selon le sinistre ?"""
    a = df.loc[df.Claim == 1, col].dropna()
    b = df.loc[df.Claim == 0, col].dropna()
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return {"median_claim": float(a.median()), "median_no_claim": float(b.median()),
            "p_value": float(p)}


def run_eda(df: pd.DataFrame, save: bool = True) -> dict:
    res: dict = {}
    n, n_claim = len(df), int(df.Claim.sum())
    res["target"] = {"n": n, "claims": n_claim, "rate_pct": round(n_claim / n * 100, 2)}

    # --- tests statistiques -------------------------------------------------
    df = df.assign(Gender_known=np.where(df.Gender == "Unknown", "Unknown", "Known"))
    res["chi2"] = {c: chi2_claim(df, c) for c in
                   ["Agency_Type", "Distribution_Channel", "Agency", "Product_Name", "Gender_known"]}
    res["mannwhitney"] = {c: mannwhitney_claim(df, c) for c in cfg.NUM_COLS}
    res["corr"] = df[cfg.NUM_COLS].corr(method="spearman").round(2).to_dict()
    res["rate_by"] = {c: claim_rate_by(df, c, 30).head(8).reset_index().to_dict("records")
                      for c in ["Agency_Type", "Agency", "Product_Name", "Gender_known"]}
    res["top_agency_share_pct"] = round(float(df.Agency.value_counts(normalize=True).iloc[0] * 100), 1)
    res["top_agency"] = df.Agency.value_counts().index[0]
    res["age_default_36_pct"] = round(float((df.Age == 36).mean() * 100), 1)

    # --- figures --------------------------------------------------------------
    # 1. cible
    fig, ax = plt.subplots(figsize=(6, 4))
    vc = df.Claim_Status.value_counts()
    ax.bar(vc.index, vc.values, color=cfg.COLORS[:2], width=0.5)
    for i, v in enumerate(vc.values):
        ax.text(i, v, f"{v:,}\n({v / n:.1%})", ha="center", va="bottom", fontweight="bold")
    ax.set_title("Variable cible : très fortement déséquilibrée")
    ax.set_ylabel("Polices"); ax.set_ylim(0, vc.max() * 1.15)
    _save(fig, "01_target.png", save); plt.close(fig)

    # 2. distributions (échelle log pour les variables très asymétriques)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, col, color in zip(axes.ravel(), cfg.NUM_COLS, cfg.COLORS):
        x = df[col].dropna()
        ax.hist(x, bins=50, color=color, alpha=0.85)
        ax.axvline(x.median(), color="black", ls=":", label=f"médiane {x.median():.0f}")
        if col != "Age":
            ax.set_yscale("log")
        ax.set_title(col + (" (axe y log)" if col != "Age" else "")); ax.legend()
    _save(fig, "02_distributions.png", save); plt.close(fig)

    # 3. numériques vs sinistre (échelle log1p pour la lisibilité)
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.5))
    for ax, col in zip(axes, cfg.NUM_COLS):
        sns.boxplot(data=df, x="Claim_Status", y=col, hue="Claim_Status",
                    palette=cfg.COLORS[:2], legend=False, ax=ax, showfliers=False)
        ax.set_title(f"{col}  (p={res['mannwhitney'][col]['p_value']:.3g})")
    fig.suptitle("Variables numériques selon le sinistre (valeurs extrêmes masquées, test de Mann-Whitney)")
    _save(fig, "03_numeric_vs_claim.png", save); plt.close(fig)

    # 4. taux de sinistre par modalité (et non comptages bruts)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, col in zip(axes, ["Agency_Type", "Gender_known", "Agency"]):
        r = claim_rate_by(df, col, 30).reset_index()
        sns.barplot(data=r, x=col, y="rate", color=cfg.COLORS[0], ax=ax)
        for i, row in r.iterrows():
            ax.text(i, row.rate, f"n={int(row.n):,}", ha="center", va="bottom", fontsize=8)
        ax.set_ylabel("Taux de sinistre (%)"); ax.tick_params(axis="x", rotation=45)
        ax.set_title(col)
    fig.suptitle("Taux de sinistre par modalité (modalités ≥ 30 polices)")
    _save(fig, "04_claim_rate_by_category.png", save); plt.close(fig)

    # 5. agences / destinations
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, col, color in zip(axes, ["Agency", "Destination"], [cfg.COLORS[0], cfg.COLORS[2]]):
        top = df[col].value_counts().head(12)[::-1]
        ax.barh(top.index, top.values, color=color)
        ax.set_title(f"Top 12 – {col}")
    _save(fig, "05_top_agency_destination.png", save); plt.close(fig)

    # 6. corrélations (Spearman : robuste aux valeurs extrêmes)
    fig, ax = plt.subplots(figsize=(6, 5))
    corr = df[cfg.NUM_COLS].corr(method="spearman")
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
                mask=np.triu(np.ones_like(corr, dtype=bool), k=1), ax=ax)
    ax.set_title("Corrélations de Spearman")
    _save(fig, "06_correlation.png", save); plt.close(fig)
    return res
