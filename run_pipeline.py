"""Point d'entrée : nettoyage -> EDA -> clustering -> modélisation.

Usage :  python run_pipeline.py
Sorties : data/processed/*.csv, reports/figures/*.png, reports/results.json
"""
import json
import time

from travel_insurance import config as cfg
from travel_insurance import clustering, data, eda, modeling


def main() -> dict:
    t0 = time.time()
    df, log = data.prepare()
    print(f"[1/4] Nettoyage  : {log}")
    res = {"cleaning": log}
    res["eda"] = eda.run_eda(df)
    print("[2/4] EDA terminée")
    df_cl, res["clustering"] = clustering.run_clustering(df)
    df_cl.to_csv(cfg.CLEAN_CSV.with_name("travel_insurance_clusters.csv"), index=False)
    print("[3/4] Clustering terminé")
    res["modeling"] = modeling.run_modeling(df)
    print("[4/4] Modélisation terminée")
    cfg.RESULTS_JSON.write_text(json.dumps(res, indent=2, ensure_ascii=False, default=str))
    print(f"Résultats -> {cfg.RESULTS_JSON}  ({time.time() - t0:.0f}s)")
    return res


if __name__ == "__main__":
    main()
