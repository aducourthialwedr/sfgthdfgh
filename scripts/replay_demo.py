"""Démo : construit le journal, rejoue l'historique, affiche l'état obtenu.

Usage :
    python scripts/generate_data.py --n-invoices 2000   # si data/ est vide
    python scripts/replay_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.events import build_journal  # noqa: E402
from src.state import LedgerState  # noqa: E402


def load_tables(data_dir: Path) -> dict[str, pd.DataFrame]:
    names = ["payment", "invoice", "imputation", "assignor", "debtor", "agreement"]
    return {name: pd.read_parquet(data_dir / f"{name}.parquet") for name in names}


def main() -> None:
    data_dir = Path("data")
    tables = load_tables(data_dir)

    print(f"Journal : dérivation à partir de {sum(len(t) for t in tables.values())} lignes source")
    journal = build_journal(tables)
    print(f"Journal : {len(journal)} événements, "
          f"de {journal[0].timestamp} à {journal[-1].timestamp}")

    by_type: dict[str, int] = {}
    for e in journal:
        by_type[e.type] = by_type.get(e.type, 0) + 1
    for t, n in sorted(by_type.items()):
        print(f"  {t:<20} {n:>6}")

    state = LedgerState()
    for event in journal:
        state.apply(event)
    print(f"\nRejeu complet terminé, horloge finale : {state.clock}")

    invoice_table = tables["invoice"].set_index("invoice_id")
    sample_ids = invoice_table.index[:5]
    print("\nComparaison état final LedgerState vs current_amount du générateur :")
    for inv_id in sample_ids:
        ledger_amount = state.current_amount(inv_id, as_of=journal[-1].timestamp)
        generator_amount = invoice_table.loc[inv_id, "current_amount"]
        status = "OK" if ledger_amount == generator_amount else "MISMATCH"
        print(f"  {inv_id}  ledger={ledger_amount:>8}  generateur={generator_amount:>8}  {status}")

    debtor_id = tables["debtor"]["party_id"].iloc[0]
    stats = state.behavioral_stats(debtor_id, as_of=journal[-1].timestamp)
    print(f"\nbehavioral_stats({debtor_id}, fin de période) :")
    for k, v in stats.items():
        print(f"  {k:<28} {v}")


if __name__ == "__main__":
    main()
