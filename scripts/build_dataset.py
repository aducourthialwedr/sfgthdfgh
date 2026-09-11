"""Construit le jeu d'entraînement (§8.3) et l'écrit en Parquet partitionné
par mois dans data/dataset/.

Usage :
    python scripts/generate_data.py --no-report   # si data/ est vide
    python scripts/build_dataset.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.replay import build_training_rows, write_partitioned_dataset  # noqa: E402


def load_tables(data_dir: Path) -> dict[str, pd.DataFrame]:
    names = ["payment", "invoice", "imputation", "assignor", "debtor", "agreement", "_technical_ibans"]
    return {name: pd.read_parquet(data_dir / f"{name}.parquet") for name in names}


def main() -> None:
    data_dir = Path("data")
    tables = load_tables(data_dir)

    t0 = time.time()
    dataset = build_training_rows(tables)
    elapsed = time.time() - t0

    out_dir = data_dir / "dataset"
    write_partitioned_dataset(dataset, out_dir)

    n_payments_covered = dataset["payment_id"].nunique()
    n_payments_total = len(tables["payment"])
    print(f"Rejeu + featurisation en {elapsed:.1f}s")
    print(f"Dataset : {len(dataset)} lignes (paires candidates)")
    print(f"Paiements couverts : {n_payments_covered}/{n_payments_total} "
          f"({n_payments_covered / n_payments_total:.1%})")
    print(f"Positifs : {dataset['label'].sum()} ({dataset['label'].mean():.2%})")
    print(f"Colonnes : {list(dataset.columns)}")

    print(f"\nÉcrit dans {out_dir}, partitionné par mois :")
    for f in sorted(out_dir.glob("month=*.parquet")):
        n = len(pd.read_parquet(f))
        print(f"  {f.name:<24} {n:>7} lignes")


if __name__ == "__main__":
    main()
