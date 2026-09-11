"""Évalue l'étage D (Phase 9, §7.1) : courbe taux d'automatisation / précision,
seuils différenciés par segment, effet de la condition de marge.

Réutilise les modèles entraînés en Phase 7 (models/*.joblib).

Usage :
    python scripts/train_model.py   # si models/ est vide
    python scripts/evaluate_decision.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.decision import (  # noqa: E402
    DEFAULT_MARGIN_DELTA,
    automation_precision_curve,
    classify,
    fit_segment_thresholds,
    top1_with_margin,
)
from src.model import add_competition_features, calibrated_scores, prepare_features, raw_scores, temporal_split  # noqa: E402

MODELS_DIR = Path("models")
TARGET_PRECISION = 0.995
AMOUNT_TIER_LABELS = ["petit", "moyen", "gros"]


def load_tables(data_dir: Path) -> dict[str, pd.DataFrame]:
    names = ["payment", "invoice", "imputation", "assignor", "debtor", "agreement", "ground_truth", "_technical_ibans"]
    return {name: pd.read_parquet(data_dir / f"{name}.parquet") for name in names}


def load_dataset(data_dir: Path) -> pd.DataFrame:
    files = sorted((data_dir / "dataset").glob("month=*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def add_amount_tier(df: pd.DataFrame, payment_amounts: pd.Series, breakpoints: tuple[float, float]) -> pd.DataFrame:
    out = df.copy()
    amounts = out["payment_id"].map(payment_amounts)
    out["amount_tier"] = pd.cut(
        amounts, bins=[-np.inf, breakpoints[0], breakpoints[1], np.inf], labels=AMOUNT_TIER_LABELS
    ).astype(str)
    return out


def main() -> None:
    data_dir = Path("data")
    tables = load_tables(data_dir)

    dataset = prepare_features(load_dataset(data_dir))
    split = temporal_split(dataset)

    pass1_model = joblib.load(MODELS_DIR / "stage_b_pass1_model.joblib")
    pass2_model = joblib.load(MODELS_DIR / "stage_b_pass2_model.joblib")
    calibrator = joblib.load(MODELS_DIR / "stage_b_pass2_calibrator.joblib")

    val2 = add_competition_features(split.val, raw_scores(pass1_model, split.val))
    test2 = add_competition_features(split.test, raw_scores(pass1_model, split.test))
    val_scores = calibrated_scores(pass2_model, calibrator, val2)
    test_scores = calibrated_scores(pass2_model, calibrator, test2)

    top1_val = top1_with_margin(val2, val_scores)
    top1_test = top1_with_margin(test2, test_scores)

    payment_amounts = tables["payment"].set_index("payment_id")["amount"]
    breakpoints = tuple(payment_amounts.quantile([0.33, 0.66]).values)
    top1_val = add_amount_tier(top1_val, payment_amounts, breakpoints)
    top1_test = add_amount_tier(top1_test, payment_amounts, breakpoints)
    print(f"Tranches de montant (33e/66e centile, sur payment.amount) : "
          f"< {breakpoints[0] / 100:.0f}€ / < {breakpoints[1] / 100:.0f}€ / au-delà")

    # --- courbe taux d'automatisation / précision ----------------------------
    targets = [0.90, 0.95, 0.98, 0.99, 0.995, 0.999]
    curve = automation_precision_curve(top1_val, top1_test, targets, segment_columns=None)
    print("\nCourbe taux d'automatisation / précision (seuil global, sans segmentation) :")
    print(curve.to_string(index=False, formatters={
        "target_precision": "{:.3%}".format,
        "automation_rate": "{:.2%}".format,
        "realized_precision": "{:.4%}".format,
    }))

    # --- seuils différenciés par segment --------------------------------------
    segment_columns = ["market", "bankroll_code", "amount_tier"]
    thresholds = fit_segment_thresholds(top1_val, segment_columns, target_precision=TARGET_PRECISION)
    print(f"\nSeuil global τ_high (précision >= {TARGET_PRECISION:.1%}) : {thresholds.global_tau_high:.4f}")
    print(f"Seuils par segment estimés ({len(thresholds.per_segment_tau_high)} segments avec assez de volume) :")
    for key, tau in sorted(thresholds.per_segment_tau_high.items(), key=lambda kv: -kv[1]):
        print(f"  {dict(zip(segment_columns, key))} -> τ_high={tau:.4f}")

    classified_segmented = classify(top1_test, thresholds, margin_delta=DEFAULT_MARGIN_DELTA)
    auto_seg = classified_segmented[classified_segmented["decision"] == "AUTO_VALIDATION"]
    print(f"\nAvec seuils par segment, sur test :")
    print(f"  précision réalisée    : {auto_seg['label'].mean():.4%}")
    print(f"  taux d'automatisation : {len(auto_seg) / len(top1_test):.4%} ({len(auto_seg)}/{len(top1_test)})")

    # --- comparaison sans segmentation (seuil unique global) ------------------
    thresholds_flat = fit_segment_thresholds(top1_val, [], target_precision=TARGET_PRECISION)
    classified_flat = classify(top1_test, thresholds_flat, margin_delta=DEFAULT_MARGIN_DELTA)
    auto_flat = classified_flat[classified_flat["decision"] == "AUTO_VALIDATION"]
    print(f"\nSans segmentation (un seul τ_high global), sur test :")
    print(f"  précision réalisée    : {auto_flat['label'].mean():.4%}")
    print(f"  taux d'automatisation : {len(auto_flat) / len(top1_test):.4%} ({len(auto_flat)}/{len(top1_test)})")

    # --- effet de la condition de marge ---------------------------------------
    classified_no_margin = classify(top1_test, thresholds, margin_delta=0.0)
    auto_no_margin = classified_no_margin[classified_no_margin["decision"] == "AUTO_VALIDATION"]
    print(f"\nSans condition de marge (delta=0), sur test :")
    print(f"  précision réalisée    : {auto_no_margin['label'].mean():.4%}")
    print(f"  taux d'automatisation : {len(auto_no_margin) / len(top1_test):.4%} ({len(auto_no_margin)}/{len(top1_test)})")

    margin_blocked = classified_segmented[
        (classified_segmented["score"] >= classified_segmented["tau_high_applied"])
        & (classified_segmented["margin"] < DEFAULT_MARGIN_DELTA)
    ]
    print(f"\nPaiements à score suffisant mais marge insuffisante (ambiguïté détectée) : {len(margin_blocked)}")
    if not margin_blocked.empty:
        print("Exemples :")
        for row in margin_blocked.head(5).itertuples():
            print(f"  {row.payment_id} -> {row.invoice_id}  score={row.score:.4f}  marge={row.margin:.4f}  "
                  f"vrai={'POSITIF' if row.label else 'négatif'}")

    # --- répartition finale ----------------------------------------------------
    print("\nRépartition des décisions (seuils par segment), test :")
    print(classified_segmented["decision"].value_counts().to_string())


if __name__ == "__main__":
    main()
