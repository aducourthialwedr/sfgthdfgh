"""Backtest en replay (Phase 10, §8.5) : rejoue les deux derniers mois
synthétiques (2025-02, 2025-03 — réservés depuis la Phase 5, jamais vus par
l'entraînement ni la validation) en mode production complet (étages A à D),
et mesure le taux d'automatisation réel du système de bout en bout.

Réutilise le modèle entraîné sur les 8 premiers mois (Phase 7) et les seuils
calibrés sur la validation (Phase 9) — ne réentraîne rien.

Usage :
    python scripts/train_model.py      # si models/ est vide
    python scripts/run_backtest.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.batch import run_backtest  # noqa: E402
from src.decision import DEFAULT_MARGIN_DELTA, DEFAULT_TAU_LOW, fit_segment_thresholds, top1_with_margin  # noqa: E402
from src.model import add_competition_features, calibrated_scores, prepare_features, raw_scores, temporal_split  # noqa: E402

MODELS_DIR = Path("models")
TARGET_PRECISION = 0.995
BACKTEST_START = pd.Timestamp("2025-02-01")
BACKTEST_END = pd.Timestamp("2025-03-31")


def load_tables(data_dir: Path) -> dict[str, pd.DataFrame]:
    names = ["payment", "invoice", "imputation", "assignor", "debtor", "agreement", "ground_truth"]
    return {name: pd.read_parquet(data_dir / f"{name}.parquet") for name in names}


def load_dataset(data_dir: Path) -> pd.DataFrame:
    files = sorted((data_dir / "dataset").glob("month=*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def main() -> None:
    data_dir = Path("data")
    tables = load_tables(data_dir)

    pass1_model = joblib.load(MODELS_DIR / "stage_b_pass1_model.joblib")
    pass2_model = joblib.load(MODELS_DIR / "stage_b_pass2_model.joblib")
    calibrator = joblib.load(MODELS_DIR / "stage_b_pass2_calibrator.joblib")

    # Seuils calibrés sur la validation (Phase 9) — le backtest utilise le
    # système déjà figé, il ne recalibre rien sur la période rejouée.
    dataset = prepare_features(load_dataset(data_dir))
    split = temporal_split(dataset)
    val2 = add_competition_features(split.val, raw_scores(pass1_model, split.val))
    val_scores = calibrated_scores(pass2_model, calibrator, val2)
    top1_val = top1_with_margin(val2, val_scores)

    # `top1_val` conserve déjà market/bankroll_code (colonnes de `val2`,
    # simplement filtré à la ligne top-1 par paiement) ; il ne manque que
    # la tranche de montant.
    payment_amounts = tables["payment"].set_index("payment_id")["amount"]
    breakpoints = tuple(payment_amounts.quantile([0.33, 0.66]).values)
    top1_val = top1_val.copy()
    top1_val["amount_tier"] = pd.cut(
        top1_val["payment_id"].map(payment_amounts),
        bins=[-np.inf, breakpoints[0], breakpoints[1], np.inf],
        labels=["petit", "moyen", "gros"],
    ).astype(str)

    thresholds = fit_segment_thresholds(
        top1_val, ["market", "bankroll_code", "amount_tier"], target_precision=TARGET_PRECISION
    )
    print(f"Seuil global τ_high (validation, précision cible {TARGET_PRECISION:.1%}) : "
          f"{thresholds.global_tau_high:.4f}")
    print(f"Segments avec seuil dédié : {len(thresholds.per_segment_tau_high)}")

    print(f"\nBacktest {BACKTEST_START.date()} -> {BACKTEST_END.date()} "
          f"(jamais vu par l'entraînement ni la validation)...")
    t0 = time.time()
    decisions, pending = run_backtest(
        tables, BACKTEST_START, BACKTEST_END,
        pass1_model, pass2_model, calibrator,
        thresholds, amount_breakpoints=breakpoints,
        margin_delta=DEFAULT_MARGIN_DELTA, tau_low=DEFAULT_TAU_LOW, retention_days=60,
    )
    elapsed = time.time() - t0
    print(f"Backtest exécuté en {elapsed:.1f}s")

    # --- comparaison aux imputations réelles ----------------------------------
    imputation = tables["imputation"]
    true_pairs = set(zip(imputation["payment_id"], imputation["invoice_id"]))
    ground_truth = tables["ground_truth"]

    payments_in_window = tables["payment"][
        (pd.to_datetime(tables["payment"]["value_date"]) >= BACKTEST_START)
        & (pd.to_datetime(tables["payment"]["value_date"]) <= BACKTEST_END)
    ]
    n_payments_window = len(payments_in_window)

    by_decision = {"AUTO_VALIDATION": [], "REVIEW": [], "REJET": []}
    for d in decisions:
        by_decision[d.decision].append(d)

    def _pairs(dec) -> set[tuple[str, str]]:
        return {(pid, iid) for pid in dec.proposal.payment_ids for iid in dec.proposal.invoice_ids}

    auto_pairs: set[tuple[str, str]] = set()
    for d in by_decision["AUTO_VALIDATION"]:
        auto_pairs |= _pairs(d)
    auto_payment_ids = {pid for d in by_decision["AUTO_VALIDATION"] for pid in d.proposal.payment_ids}

    n_correct_pairs = sum(1 for pair in auto_pairs if pair in true_pairs)
    precision = n_correct_pairs / len(auto_pairs) if auto_pairs else float("nan")
    automation_rate = len(auto_payment_ids) / n_payments_window if n_payments_window else float("nan")

    print("\n" + "=" * 70)
    print("BACKTEST — Phase 10 (§8.5) : le système au complet, deux mois jamais vus")
    print("=" * 70)
    print(f"Paiements dans la fenêtre de backtest      : {n_payments_window}")
    print(f"Décisions AUTO_VALIDATION (paiements)      : {len(auto_payment_ids)}")
    print(f"Décisions REVIEW (paiements)                : "
          f"{len({pid for d in by_decision['REVIEW'] for pid in d.proposal.payment_ids})}")
    print(f"Décisions REJET (paiements)                 : "
          f"{len({pid for d in by_decision['REJET'] for pid in d.proposal.payment_ids})}")
    print(f"Reliquat non résolu en fin de période        : {len(pending)}")
    print()
    print(f"PRÉCISION réelle (paires auto-validées correctes) : {precision:.4%}")
    print(f"TAUX D'AUTOMATISATION réel (bout en bout)          : {automation_rate:.4%}")

    print("\nRépartition par type de cas (ground_truth), parmi les auto-validés :")
    gt_window = ground_truth[ground_truth["payment_id"].isin(payments_in_window["payment_id"])]
    for case_type, group in gt_window.drop_duplicates("payment_id").groupby("case_type"):
        pids = set(group["payment_id"])
        n_total = len(pids)
        n_auto = len(pids & auto_payment_ids)
        print(f"  {case_type:<15} {n_auto:>5}/{n_total:<5} ({n_auto / n_total:.2%} auto-validés)")

    if by_decision["AUTO_VALIDATION"]:
        wrong = [d for d in by_decision["AUTO_VALIDATION"] if not _pairs(d).issubset(true_pairs)]
        print(f"\nGroupes auto-validés incorrects : {len(wrong)}/{len(by_decision['AUTO_VALIDATION'])}")
        for d in wrong[:10]:
            print(f"  {d.date.date()} {d.proposal.payment_ids} -> {d.proposal.invoice_ids}  "
                  f"score={d.score:.4f} marge={d.margin:.4f}")

    print("=" * 70)


if __name__ == "__main__":
    main()
