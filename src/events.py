"""Journal d'événements dérivé des six tables (§8.1).

Un journal est une liste d'`Event` triée de façon totalement déterministe :
d'abord par timestamp, puis par type d'événement (priorité fixe), puis par
identifiant. Cette règle de départage est nécessaire car beaucoup de champs
source sont des dates sans heure — plusieurs événements partagent alors le
même timestamp.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pandas as pd

# Priorité de départage à timestamp égal : les événements de création /
# ouverture précèdent les événements de consommation, qui précèdent les
# événements de clôture. Décision du générateur, non imposée par la spec.
EVENT_TYPE_PRIORITY: dict[str, int] = {
    "PARTY_OPENED": 0,
    "AGREEMENT_CREATED": 1,
    "INVOICE_CREATED": 2,
    "PAYMENT_RECEIVED": 3,
    "IMPUTATION_APPLIED": 4,
    "AGREEMENT_DISABLED": 5,
    "PARTY_CLOSED": 6,
}


@dataclasses.dataclass(frozen=True)
class Event:
    """Un événement du journal.

    `sort_key` = (priorité du type, tuple d'identifiants) — combiné à
    `timestamp` par `build_journal`, il rend le tri total et reproductible.
    """

    timestamp: pd.Timestamp
    type: str
    sort_key: tuple
    data: dict[str, Any]


def _events_from_records(
    records: list[dict],
    event_type: str,
    ts_field: str,
    key_fields: tuple[str, ...],
) -> list[Event]:
    priority = EVENT_TYPE_PRIORITY[event_type]
    events = []
    for r in records:
        ts = r[ts_field]
        if pd.isna(ts):
            continue
        key = tuple(r[f] for f in key_fields)
        events.append(
            Event(timestamp=pd.Timestamp(ts), type=event_type, sort_key=(priority, key), data=r)
        )
    return events


def build_journal(tables: dict[str, pd.DataFrame]) -> list[Event]:
    """Dérive le journal d'événements ordonné à partir des six tables."""
    events: list[Event] = []

    events += _events_from_records(
        tables["debtor"].to_dict("records"), "PARTY_OPENED", "opened_at", ("party_id",)
    )
    # Pas de PARTY_CLOSED pour debtor : la table n'a pas de `closed_at`
    # (schéma réel) — un débiteur n'est jamais formellement fermé.
    events += _events_from_records(
        tables["assignor"].to_dict("records"), "PARTY_OPENED", "opened_at", ("party_id",)
    )
    events += _events_from_records(
        tables["assignor"].to_dict("records"), "PARTY_CLOSED", "closed_at", ("party_id",)
    )
    events += _events_from_records(
        tables["agreement"].to_dict("records"),
        "AGREEMENT_CREATED",
        "created_at",
        ("agreement_id",),
    )
    events += _events_from_records(
        tables["agreement"].to_dict("records"),
        "AGREEMENT_DISABLED",
        "disabled_at",
        ("agreement_id",),
    )
    events += _events_from_records(
        tables["invoice"].to_dict("records"), "INVOICE_CREATED", "creation_date", ("invoice_id",)
    )
    events += _events_from_records(
        tables["payment"].to_dict("records"), "PAYMENT_RECEIVED", "value_date", ("payment_id",)
    )
    events += _events_from_records(
        tables["imputation"].to_dict("records"),
        "IMPUTATION_APPLIED",
        "updated_at",
        ("payment_id", "invoice_id"),
    )

    events.sort(key=lambda e: (e.timestamp, e.sort_key))
    return events
