"""Chargement et nettoyage des données.

Choix de nettoyage (tous journalisés dans le dictionnaire retourné) :

* ligne entièrement vide  -> supprimée
* doublons exacts         -> supprimés (hypothèse : erreurs de saisie / import)
* Duration <= 0           -> supprimée (durée de voyage impossible)
* Age > 100               -> remplacé par NaN (118 = valeur « bouchon »)
* Gender manquant         -> conservé comme catégorie « Unknown »
  (79 % de manquants : l'imputer par le mode fabriquerait de la donnée,
  et l'absence de genre est elle-même informative, cf. notebook).
* Les valeurs aberrantes ne sont PAS modifiées ici : la winsorisation est
  appliquée uniquement là où elle est utile (clustering), jamais sur le
  fichier nettoyé, afin de ne pas perdre d'information.
"""
from __future__ import annotations

import pandas as pd

from . import config as cfg


def load_raw(path=cfg.RAW_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = cfg.COLUMN_NAMES
    return df


def clean(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Retourne (dataframe nettoyé, journal des opérations)."""
    log: dict = {"n_raw": int(len(df_raw))}
    df = df_raw.copy()

    # 1. lignes vides (cible et variables numériques toutes manquantes)
    empty = df[["Claim_Status", *cfg.NUM_COLS]].isna().all(axis=1)
    log["n_empty_rows"] = int(empty.sum())
    df = df[~empty]

    # 2. doublons exacts
    log["n_duplicates"] = int(df.duplicated().sum())
    df = df.drop_duplicates()

    # 3. durées impossibles
    bad_duration = df["Duration"] <= 0
    log["n_invalid_duration"] = int(bad_duration.sum())
    df = df[~bad_duration]

    # 4. âges « bouchon »
    bad_age = df["Age"] > 100
    log["n_invalid_age"] = int(bad_age.sum())
    df = df.assign(Age=df["Age"].mask(bad_age))

    # 5. genre manquant -> catégorie explicite
    log["gender_missing_pct"] = round(float(df["Gender"].isna().mean() * 100), 2)
    df = df.assign(Gender=df["Gender"].fillna("Unknown"))

    # 6. cible binaire
    df = df.assign(Claim=(df["Claim_Status"] == "Yes").astype(int))

    log["n_clean"] = int(len(df))
    log["age_missing_after"] = int(df["Age"].isna().sum())
    return df.reset_index(drop=True), log


def prepare(save: bool = True) -> tuple[pd.DataFrame, dict]:
    df, log = clean(load_raw())
    if save:
        cfg.CLEAN_CSV.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cfg.CLEAN_CSV, index=False)
    return df, log
