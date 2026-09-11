"""Évalue la baseline par règles (Phase 3) : taux de couverture et précision.

C'est le chiffre à battre — tous les modèles ultérieurs (étage B) seront
comparés à ces nombres.

Usage :
    python scripts/generate_data.py --no-report   # si data/ est vide
    python scripts/evaluate_baseline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baseline import compute_baseline_predictions  # noqa: E402


def load_tables(data_dir: Path) -> dict[str, pd.DataFrame]:
    names = ["payment", "invoice", "imputation", "assignor", "debtor", "agreement", "ground_truth", "_technical_ibans"]
    return {name: pd.read_parquet(data_dir / f"{name}.parquet") for name in names}


def main() -> None:
    data_dir = Path("data")
    tables = load_tables(data_dir)

    ground_truth = tables["ground_truth"]
    true_pairs = set(zip(ground_truth["payment_id"], ground_truth["invoice_id"]))
    matchable_payment_ids = set(ground_truth["payment_id"])

    predictions = compute_baseline_predictions(tables)

    n_total_payments = len(tables["payment"])
    n_matchable = len(matchable_payment_ids)
    n_proposed = len(predictions)
    correct = {
        pid: inv_id for pid, inv_id in predictions.items() if (pid, inv_id) in true_pairs
    }
    n_correct = len(correct)
    incorrect = {pid: inv_id for pid, inv_id in predictions.items() if pid not in correct}

    precision = n_correct / n_proposed if n_proposed else float("nan")
    coverage = n_correct / n_matchable if n_matchable else float("nan")

    print("=" * 70)
    print("BASELINE PAR RÈGLES — Phase 3 (chiffre à battre)")
    print("=" * 70)
    print(f"Paiements totaux              : {n_total_payments}")
    print(f"Paiements avec vraie imputation: {n_matchable}")
    print(f"Propositions de la baseline    : {n_proposed}")
    print(f"Propositions correctes         : {n_correct}")
    print(f"Propositions incorrectes       : {len(incorrect)}")
    print()
    print(f"PRÉCISION  (correct / proposé)  : {precision:.4%}")
    print(f"COUVERTURE (correct / matchable): {coverage:.4%}")

    print("\nCouverture par type de cas (ground_truth) :")
    for case_type, group in ground_truth.drop_duplicates("payment_id").groupby("case_type"):
        pids = set(group["payment_id"])
        n_case = len(pids)
        n_case_correct = sum(1 for pid in pids if pid in correct)
        print(f"  {case_type:<15} {n_case_correct:>6}/{n_case:<6} ({n_case_correct / n_case:.2%})")

    if incorrect:
        print(f"\n{len(incorrect)} propositions incorrectes, exemples :")
        for pid, inv_id in list(incorrect.items())[:10]:
            true_invs = ground_truth[ground_truth["payment_id"] == pid]["invoice_id"].tolist()
            print(f"  {pid} -> proposé {inv_id}, vrai(s) {true_invs or '(aucune imputation réelle)'}")

    print("=" * 70)


if __name__ == "__main__":
    main()
