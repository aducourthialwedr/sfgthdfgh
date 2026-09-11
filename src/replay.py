"""Rejeu de l'historique et construction du jeu d'entraînement (§8.3).

Même code que la production (§8.4) rejouera plus tard sur le présent : le
curseur avance sur le journal, les features sont extraites *avant*
`state.apply(event)`, jamais après. Le labelling est une seconde passe,
séparée, sans accès à l'état pendant le rejeu — la vérité terrain
(`imputation`) n'est consultée qu'une fois le dataset de features déjà
entièrement construit.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.blocking import build_static_lookups, generate_candidates
from src.events import build_journal
from src.features import featurize
from src.state import LedgerState

_FRONT_COLUMNS = ["payment_id", "invoice_id", "t", "label"]


def build_training_rows(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Rejoue le journal dérivé de `tables`, produit une ligne de features
    par paire candidate `(payment, invoice)`, colonne `t` = timestamp du
    paiement. Aucune ligne n'est écrite pour un paiement sans candidat."""
    journal = build_journal(tables)
    lookups = build_static_lookups(tables)

    state = LedgerState()
    rows: list[dict] = []
    for event in journal:
        if event.type == "PAYMENT_RECEIVED":
            payment = event.data
            as_of = event.timestamp
            candidate_ids = generate_candidates(payment, state, as_of, lookups)
            for invoice_id in candidate_ids:
                invoice = state.get_invoice(invoice_id, as_of=as_of)
                feats = featurize(payment, invoice, state, as_of, lookups)
                feats["payment_id"] = payment["payment_id"]
                feats["invoice_id"] = invoice_id
                feats["t"] = as_of
                rows.append(feats)
        state.apply(event)  # après extraction, jamais avant (§8.3)

    dataset = pd.DataFrame(rows)
    return _label_dataset(dataset, tables["imputation"])


def _label_dataset(dataset: pd.DataFrame, imputation: pd.DataFrame) -> pd.DataFrame:
    """Seconde passe : label positif si la paire figure dans `imputation`.
    N'a accès qu'au dataset de features déjà construit, jamais à l'état."""
    if dataset.empty:
        dataset["label"] = pd.Series(dtype=bool)
        return dataset
    true_pairs = set(zip(imputation["payment_id"], imputation["invoice_id"]))
    dataset["label"] = [
        (pid, iid) in true_pairs for pid, iid in zip(dataset["payment_id"], dataset["invoice_id"])
    ]
    ordered = _FRONT_COLUMNS + [c for c in dataset.columns if c not in _FRONT_COLUMNS]
    return dataset[ordered]


def write_partitioned_dataset(dataset: pd.DataFrame, out_dir: Path) -> None:
    """Écrit le dataset en Parquet, partitionné par mois de `t`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old_file in out_dir.glob("month=*.parquet"):
        old_file.unlink()
    month = pd.to_datetime(dataset["t"]).dt.to_period("M").astype(str)
    for period, group in dataset.groupby(month):
        group.to_parquet(out_dir / f"month={period}.parquet", index=False)
