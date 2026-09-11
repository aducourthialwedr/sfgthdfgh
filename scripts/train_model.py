"""Entraîne l'étage B et rapporte les métriques du §5.4 : passe unique
(Phases 5-6) vs deux passes avec features de compétition (Phase 7),
comparées entre elles et à la baseline de la Phase 3, sur le même test set.

Usage :
    python scripts/generate_data.py --no-report   # si data/ est vide
    python scripts/build_dataset.py               # si data/dataset/ est vide
    python scripts/train_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baseline import compute_baseline_predictions  # noqa: E402
from src.model import (  # noqa: E402
    automation_rate_at_threshold,
    automation_threshold_for_precision,
    calibrated_scores,
    fit_calibrator,
    precision_at_1_and_mrr,
    prepare_features,
    raw_scores,
    temporal_split,
    train_model,
    train_two_pass,
    worst_errors,
)

MODELS_DIR = Path("models")
TARGET_PRECISION = 0.995


def load_tables(data_dir: Path) -> dict[str, pd.DataFrame]:
    names = ["payment", "invoice", "imputation", "assignor", "debtor", "agreement", "ground_truth", "_technical_ibans"]
    return {name: pd.read_parquet(data_dir / f"{name}.parquet") for name in names}


def load_dataset(data_dir: Path) -> pd.DataFrame:
    files = sorted((data_dir / "dataset").glob("month=*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def evaluate(
    label: str,
    model,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    ground_truth: pd.DataFrame,
    payment_ids_1to1: set[str],
) -> dict:
    """Calibre, évalue (§5.4) et imprime le rapport pour un modèle donné.
    Retourne les chiffres clés pour la comparaison finale."""
    calibrator = fit_calibrator(model, val_df)
    val_scores = calibrated_scores(model, calibrator, val_df)
    test_scores = calibrated_scores(model, calibrator, test_df)
    test_scores_raw = raw_scores(model, test_df)

    test_payment_ids = set(test_df["payment_id"])
    gt_test = ground_truth[ground_truth["payment_id"].isin(test_payment_ids)]
    found_pairs = set(zip(test_df["payment_id"], test_df["invoice_id"]))
    recall_at_blocking = (
        sum(1 for r in gt_test.itertuples() if (r.payment_id, r.invoice_id) in found_pairs)
        / len(gt_test)
        if len(gt_test)
        else float("nan")
    )
    precision_at_1, mrr = precision_at_1_and_mrr(test_df, test_scores, payment_ids_1to1)
    auc = roc_auc_score(test_df["label"], test_scores_raw)  # séparation des scores, score brut

    tau = automation_threshold_for_precision(val_df, val_scores, target_precision=TARGET_PRECISION)
    n_test_payments = test_df["payment_id"].nunique()
    automation = (
        automation_rate_at_threshold(test_df, test_scores, tau, n_test_payments) if tau is not None else None
    )

    print("\n" + "-" * 70)
    print(label)
    print("-" * 70)
    print(f"AUC (séparation des scores, test)     : {auc:.4f}")
    print(f"recall@blocking (test)                : {recall_at_blocking:.4%}")
    print(f"precision@1 (1↔1, test)                : {precision_at_1:.4%}")
    print(f"MRR (1↔1, test)                        : {mrr:.4f}")
    if automation is not None:
        print(f"Seuil calibré pour précision >= {TARGET_PRECISION:.1%} (sur val) : {tau:.4f}")
        print(f"  -> précision réalisée sur test       : {automation['precision']:.4%}")
        print(
            f"  -> taux d'automatisation sur test     : {automation['automation_rate']:.4%} "
            f"({automation['n_selected']}/{n_test_payments} paiements)"
        )
    else:
        print(f"Aucun seuil n'atteint {TARGET_PRECISION:.1%} de précision sur la validation.")

    return dict(
        label=label,
        model=model,
        calibrator=calibrator,
        auc=auc,
        precision_at_1=precision_at_1,
        mrr=mrr,
        automation=automation,
        test_scores=test_scores,
    )


def main() -> None:
    data_dir = Path("data")
    tables = load_tables(data_dir)
    ground_truth = tables["ground_truth"]

    dataset = prepare_features(load_dataset(data_dir))
    split = temporal_split(dataset)
    print(
        f"Split : train {len(split.train)} lignes / {split.train['payment_id'].nunique()} paiements "
        f"({split.train['t'].min().date()} -> {split.train['t'].max().date()})"
    )
    print(
        f"        val   {len(split.val)} lignes / {split.val['payment_id'].nunique()} paiements "
        f"({split.val['t'].min().date()} -> {split.val['t'].max().date()})"
    )
    print(
        f"        test  {len(split.test)} lignes / {split.test['payment_id'].nunique()} paiements "
        f"({split.test['t'].min().date()} -> {split.test['t'].max().date()})"
    )

    case_type_by_payment = ground_truth.drop_duplicates("payment_id").set_index("payment_id")["case_type"]
    payment_ids_1to1 = set(
        case_type_by_payment[case_type_by_payment.isin(["1-1_clean", "1-1_noisy"])].index
    )

    print("\nEntraînement du modèle de base (passe unique)...")
    pass1_model = train_model(split.train, split.val)
    result1 = evaluate(
        "PASSE UNIQUE (Phases 5-6)", pass1_model, split.val, split.test, ground_truth, payment_ids_1to1
    )

    print("\nEntraînement deuxième passe (features de compétition, §5.3)...")
    two_pass = train_two_pass(split, pass1_model=pass1_model)
    result2 = evaluate(
        "DEUX PASSES (Phase 7)", two_pass.pass2_model, two_pass.val2, two_pass.test2,
        ground_truth, payment_ids_1to1,
    )

    MODELS_DIR.mkdir(exist_ok=True)
    two_pass.pass2_model.booster_.save_model(str(MODELS_DIR / "stage_b_lightgbm.txt"))
    joblib.dump(two_pass.pass2_model, MODELS_DIR / "stage_b_pass2_model.joblib")
    joblib.dump(result2["calibrator"], MODELS_DIR / "stage_b_pass2_calibrator.joblib")
    joblib.dump(pass1_model, MODELS_DIR / "stage_b_pass1_model.joblib")

    # --- baseline sur le même test set ---------------------------------------
    test_payment_ids = set(split.test["payment_id"])
    gt_test = ground_truth[ground_truth["payment_id"].isin(test_payment_ids)]
    baseline_predictions = compute_baseline_predictions(tables)
    true_pairs = set(zip(tables["imputation"]["payment_id"], tables["imputation"]["invoice_id"]))
    baseline_test = {pid: inv for pid, inv in baseline_predictions.items() if pid in test_payment_ids}
    baseline_correct = sum(1 for pid, inv in baseline_test.items() if (pid, inv) in true_pairs)
    baseline_precision = baseline_correct / len(baseline_test) if baseline_test else float("nan")
    baseline_coverage = baseline_correct / len(gt_test["payment_id"].unique()) if len(gt_test) else float("nan")

    # --- comparaison finale -----------------------------------------------------
    print("\n" + "=" * 70)
    print("COMPARAISON — baseline vs passe unique vs deux passes (même test set)")
    print("=" * 70)
    header = f"{'':<28}{'précision':>12}{'automatisation':>16}{'AUC':>10}"
    print(header)
    print(f"{'Baseline (règles)':<28}{baseline_precision:>12.2%}{baseline_coverage:>16.2%}{'':>10}")
    for r in (result1, result2):
        auto = r["automation"]
        precision_str = f"{auto['precision']:.2%}" if auto else "n/a"
        automation_str = f"{auto['automation_rate']:.2%}" if auto else "n/a"
        print(f"{r['label']:<28}{precision_str:>12}{automation_str:>16}{r['auc']:>10.4f}")

    n_test_payments = split.test["payment_id"].nunique()
    for r in (result1, result2):
        auto = r["automation"]
        if auto is None:
            continue
        cost_fp, cost_review = 10, 1
        n_fp = auto["n_selected"] - auto["n_correct"]
        n_review = n_test_payments - auto["n_selected"]
        r["cost"] = n_fp * cost_fp + n_review * cost_review
    print("\nCoût illustratif (faux positif = 10x coût revue manuelle) :")
    baseline_n_fp = len(baseline_test) - baseline_correct
    baseline_n_review = n_test_payments - len(baseline_test)
    print(f"  baseline               : {baseline_n_fp} FP + {baseline_n_review} revues "
          f"-> coût {baseline_n_fp * 10 + baseline_n_review}")
    for r in (result1, result2):
        if "cost" in r:
            print(f"  {r['label']:<22} : coût {r['cost']}")

    # --- importance des features (deuxième passe) --------------------------------
    importances = pd.Series(
        two_pass.pass2_model.booster_.feature_importance(importance_type="gain"),
        index=two_pass.pass2_model.booster_.feature_name(),
    ).sort_values(ascending=False)
    print("\nImportance des features, deuxième passe (gain, top 15) :")
    for name, value in importances.head(15).items():
        print(f"  {name:<32} {value:>12.1f}")

    # --- dix pires erreurs (deuxième passe) ---------------------------------------
    print("\nDix pires erreurs, deuxième passe (test) :")
    errs = worst_errors(two_pass.test2, result2["test_scores"], n=10)
    invoice_ref = tables["invoice"].set_index("invoice_id")["client_reference"]
    payment_label = tables["payment"].set_index("payment_id")["label"]
    for row in errs.itertuples():
        true_label = "POSITIF" if row.label else "négatif"
        print(
            f"  {row.payment_id} <-> {row.invoice_id}  score={row.score:.4f}  vrai={true_label}  "
            f"rang={row.rank_in_payment}  marge={row.score_margin_to_second:.4f}  "
            f"n_cand={row.n_candidates}  libellé={payment_label.get(row.payment_id, '')!r}  "
            f"ref_facture={invoice_ref.get(row.invoice_id, '')!r}"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()
