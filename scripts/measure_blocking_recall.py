"""Mesure le rappel de l'étage A (blocking) contre `ground_truth.parquet`.

Rejoue le journal chronologiquement ; pour chaque paiement, génère les
candidats *avant* d'appliquer l'événement à l'état (même discipline que le
rejeu de production, §8.3), puis compare a posteriori aux paires réellement
imputées.

Usage :
    python scripts/generate_data.py --no-report   # si data/ est vide
    python scripts/measure_blocking_recall.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.blocking import build_static_lookups, generate_candidates  # noqa: E402
from src.events import build_journal  # noqa: E402
from src.state import LedgerState  # noqa: E402


def load_tables(data_dir: Path) -> dict[str, pd.DataFrame]:
    names = ["payment", "invoice", "imputation", "assignor", "debtor", "agreement", "ground_truth"]
    return {name: pd.read_parquet(data_dir / f"{name}.parquet") for name in names}


def main() -> None:
    data_dir = Path("data")
    tables = load_tables(data_dir)
    journal = build_journal(tables)
    lookups = build_static_lookups(tables)

    state = LedgerState()
    candidates_by_payment: dict[str, set[str]] = {}

    t0 = time.time()
    n_payments = 0
    for event in journal:
        if event.type == "PAYMENT_RECEIVED":
            payment = event.data
            candidates = generate_candidates(
                payment,
                state,
                as_of=event.timestamp,
                debtor_by_iban=lookups["debtor_by_iban"],
                assignor_by_iban=lookups["assignor_by_iban"],
                debtor_name_tokens=lookups["debtor_name_tokens"],
            )
            candidates_by_payment[payment["payment_id"]] = set(candidates)
            n_payments += 1
        state.apply(event)
    elapsed = time.time() - t0
    print(f"Blocking exécuté sur {n_payments} paiements en {elapsed:.1f}s "
          f"({1000 * elapsed / max(n_payments, 1):.2f} ms/paiement)")

    # --- Rappel global -----------------------------------------------------
    ground_truth = tables["ground_truth"]
    hits = 0
    misses = []
    for row in ground_truth.itertuples(index=False):
        cands = candidates_by_payment.get(row.payment_id, set())
        if row.invoice_id in cands:
            hits += 1
        else:
            misses.append((row.payment_id, row.invoice_id, row.case_type))
    recall = hits / len(ground_truth) if len(ground_truth) else float("nan")
    print(f"\nRappel de blocking global : {hits}/{len(ground_truth)} = {recall:.4%}")

    # --- Rappel par type de cas ---------------------------------------------
    print("\nRappel par type de cas :")
    gt_by_case = ground_truth.groupby("case_type")
    miss_df = pd.DataFrame(misses, columns=["payment_id", "invoice_id", "case_type"])
    for case_type, group in gt_by_case:
        n_total = len(group)
        n_missed = (miss_df["case_type"] == case_type).sum() if not miss_df.empty else 0
        n_hit = n_total - n_missed
        print(f"  {case_type:<15} {n_hit:>6}/{n_total:<6} ({n_hit / n_total:.2%})")

    # --- Distribution du nombre de candidats par paiement --------------------
    sizes = np.array([len(v) for v in candidates_by_payment.values()])
    print("\nCandidats par paiement :")
    print(f"  médiane : {np.median(sizes):.0f}")
    print(f"  moyenne : {sizes.mean():.1f}")
    print(f"  p90     : {np.percentile(sizes, 90):.0f}")
    print(f"  p99     : {np.percentile(sizes, 99):.0f}")
    print(f"  max     : {sizes.max()}")

    # --- Exemples de paires manquées -----------------------------------------
    if not miss_df.empty:
        print(f"\n{len(miss_df)} paires manquées, exemples :")
        print(miss_df.head(15).to_string(index=False))

    if recall < 0.99:
        print("\n/!\\ Rappel sous la cible de 99% — ne pas passer à l'étage suivant.")
    else:
        print("\nRappel >= 99% : cible atteinte.")


if __name__ == "__main__":
    main()
