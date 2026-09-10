"""CLI : génère le jeu de données synthétique et écrit les tables en Parquet.

Usage :
    python scripts/generate_data.py
    python scripts/generate_data.py --seed 7 --n-invoices 5000 --out-dir data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.generator import GeneratorParams, generate  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-debtors", type=int, default=200)
    p.add_argument("--n-assignors", type=int, default=30)
    p.add_argument("--n-invoices", type=int, default=20_000)
    p.add_argument("--n-months", type=int, default=14)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=str, default="data")
    p.add_argument("--no-report", action="store_true", help="ne pas afficher le rapport")
    return p.parse_args()


def print_report(tables: dict) -> None:
    payment = tables["payment"]
    invoice = tables["invoice"]
    ground_truth = tables["ground_truth"]

    print("=" * 70)
    print("RAPPORT DE GÉNÉRATION")
    print("=" * 70)
    print(f"Débiteurs      : {len(tables['debtor'])}")
    print(f"Cédants        : {len(tables['assignor'])}")
    print(f"Agreements     : {len(tables['agreement'])}")
    print(f"Factures       : {len(invoice)}")
    print(f"Paiements      : {len(payment)}")
    print(f"Imputations    : {len(tables['imputation'])}")
    print()

    print("Répartition des cas (ground_truth, par groupe) :")
    if not ground_truth.empty:
        n_groups = ground_truth["group_id"].nunique()
        by_case = ground_truth.drop_duplicates("group_id")["case_type"].value_counts()
        for case_type, count in by_case.items():
            pct = 100 * count / n_groups
            print(f"  {case_type:<15} {count:>6}  ({pct:5.1f} %)")
    n_matched_payments = payment["payment_id"].isin(ground_truth["payment_id"]).sum()
    n_orphan_payments = len(payment) - n_matched_payments
    pct_orphan = 100 * n_orphan_payments / len(payment) if len(payment) else 0.0
    print(f"  {'no_invoice':<15} {n_orphan_payments:>6}  ({pct_orphan:5.1f} %)  [paiements orphelins]")
    print()

    print("Exemples de libellés (bruts) :")
    sample = payment.sample(n=min(12, len(payment)), random_state=0)
    for _, row in sample.iterrows():
        label = row["label"] if row["label"] else "<vide>"
        print(f"  [{row['payment_id']}] {label!r}")
    print()

    empty_labels = (payment["label"] == "").sum()
    numeric_only = payment["label"].str.replace(r"\D", "", regex=True).eq(payment["label"]) & (
        payment["label"] != ""
    )
    print(f"Libellés vides       : {empty_labels} ({100 * empty_labels / len(payment):.1f} %)")
    print(f"Libellés numériques  : {numeric_only.sum()} ({100 * numeric_only.sum() / len(payment):.1f} %)")
    print("=" * 70)


def main() -> None:
    args = _parse_args()
    params = GeneratorParams(
        n_debtors=args.n_debtors,
        n_assignors=args.n_assignors,
        n_invoices=args.n_invoices,
        n_months=args.n_months,
        seed=args.seed,
    )
    tables = generate(params)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_parquet(out_dir / f"{name}.parquet", index=False)

    print(f"Données écrites dans {out_dir.resolve()}")
    if not args.no_report:
        print_report(tables)


if __name__ == "__main__":
    main()
