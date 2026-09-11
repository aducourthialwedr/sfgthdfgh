"""Orchestration de production (§8.4) et backtest en replay (§8.5).

Réutilise intégralement les étages A-D déjà implémentés (blocking,
features, modèle, résolution d'ensembles, décision) — ce module ne fait
qu'orchestrer, au rythme d'un batch quotidien.

Différence structurelle avec le rejeu d'entraînement (`replay.py`) : là où
l'entraînement avance l'état paiement par paiement (pour maximiser le
réalisme historique), la production avance l'état **une fois par jour**,
et traite tous les paiements du jour (nouveaux + reliquat) ensemble — sinon
deux paiements du même débiteur arrivés le même jour ne pourraient jamais
être arbitrés l'un contre l'autre (§8.4, étape 4).
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np
import pandas as pd

from src.blocking import build_static_lookups, generate_candidates, resolve_iban
from src.decision import SegmentThresholds, top1_with_margin
from src.events import build_journal
from src.features import featurize
from src.model import add_competition_features, calibrated_scores, prepare_features, raw_scores
from src.state import LedgerState
from src.subsets import Candidate, GroupProposal, propose_groups, resolve_conflicts

DEFAULT_RETENTION_DAYS = 60


@dataclasses.dataclass
class DecidedGroup:
    date: pd.Timestamp
    proposal: GroupProposal
    decision: str  # AUTO_VALIDATION / REVIEW / REJET
    score: float
    margin: float


def _amount_tier(amount_cents: int, breakpoints: tuple[float, float]) -> str:
    if amount_cents < breakpoints[0]:
        return "petit"
    if amount_cents < breakpoints[1]:
        return "moyen"
    return "gros"


def run_backtest(
    tables: dict[str, pd.DataFrame],
    backtest_start: pd.Timestamp,
    backtest_end: pd.Timestamp,
    pass1_model,
    pass2_model,
    calibrator,
    thresholds: SegmentThresholds,
    amount_breakpoints: tuple[float, float],
    margin_delta: float = 0.05,
    tau_low: float = 0.02,
    retention_days: int = DEFAULT_RETENTION_DAYS,
) -> tuple[list[DecidedGroup], dict[str, pd.Timestamp]]:
    """Rejoue `[backtest_start, backtest_end]` en mode production complet
    (étages A à D, seuils inclus) — §8.5.

    Retourne `(décisions, reliquat_non_résolu_en_fin_de_période)`.
    """
    journal = build_journal(tables)
    lookups = build_static_lookups(tables)
    payment_raw = tables["payment"].set_index("payment_id")
    invoice_agreement = tables["invoice"].set_index("invoice_id")["agreement_id"]
    invoice_debtor = tables["invoice"].set_index("invoice_id")["debtor_id"]
    agreement_market = tables["agreement"].set_index("agreement_id")["market"]

    state = LedgerState()
    journal_idx = 0
    while journal_idx < len(journal) and journal[journal_idx].timestamp < backtest_start:
        state.apply(journal[journal_idx])
        journal_idx += 1

    pending: dict[str, pd.Timestamp] = {}
    decisions: list[DecidedGroup] = []

    current_day = pd.Timestamp(backtest_start.date())
    end_day = pd.Timestamp(backtest_end.date())
    while current_day <= end_day:
        next_day = current_day + pd.Timedelta(days=1)

        # Ingestion + avance d'état jusqu'à la veille incluse (§8.4, étapes 1-2).
        while journal_idx < len(journal) and journal[journal_idx].timestamp < current_day:
            state.apply(journal[journal_idx])
            journal_idx += 1
        as_of = current_day

        for event in journal:
            if event.type == "PAYMENT_RECEIVED" and current_day <= event.timestamp < next_day:
                pid = event.data["payment_id"]
                pending.setdefault(pid, current_day)

        if pending:
            candidates_by_payment: dict[str, list[Candidate]] = {}
            rows = []
            for payment_id in pending:
                payment = payment_raw.loc[payment_id].to_dict()
                payment["payment_id"] = payment_id
                candidate_ids = generate_candidates(payment, state, as_of, lookups)
                for invoice_id in candidate_ids:
                    invoice = state.get_invoice(invoice_id, as_of=as_of)
                    feats = featurize(payment, invoice, state, as_of, lookups)
                    feats["payment_id"] = payment_id
                    feats["invoice_id"] = invoice_id
                    feats["decision_current_amount"] = int(invoice["current_amount"])
                    rows.append(feats)

            if rows:
                df = prepare_features(pd.DataFrame(rows))
                pass1_scores = raw_scores(pass1_model, df)
                df2 = add_competition_features(df, pass1_scores)
                df2["score"] = calibrated_scores(pass2_model, calibrator, df2)
                margin_by_payment = (
                    top1_with_margin(df2, df2["score"].to_numpy())
                    .set_index("payment_id")["margin"]
                )

                for row in df2.itertuples():
                    candidates_by_payment.setdefault(row.payment_id, []).append(
                        Candidate(row.invoice_id, row.decision_current_amount, float(row.score))
                    )

                payments_list = []
                for payment_id, cands in candidates_by_payment.items():
                    raw = payment_raw.loc[payment_id]
                    route, direct_debtor_id = resolve_iban(
                        dict(iban_debtor=raw["iban_debtor"]), lookups,
                    )
                    debtor_id = direct_debtor_id
                    payments_list.append(
                        dict(
                            payment_id=payment_id,
                            debtor_id=debtor_id or "UNKNOWN",
                            value_date=pd.Timestamp(raw["value_date"]),
                            amount=int(raw["amount"]),
                        )
                    )

                proposals = propose_groups(payments_list, candidates_by_payment)
                accepted, _rejected = resolve_conflicts(proposals)

                for proposal in accepted:
                    margins = [margin_by_payment.get(pid, 0.0) for pid in proposal.payment_ids]
                    margin = min(margins) if margins else 0.0

                    primary_invoice = proposal.invoice_ids[0]
                    agreement_id = invoice_agreement.get(primary_invoice)
                    market = agreement_market.get(agreement_id) if agreement_id is not None else None
                    primary_payment_amount = int(payment_raw.loc[proposal.payment_ids[0], "amount"])
                    # bankroll_code n'existe pas sur payment (schéma réel) :
                    # résolu depuis le débiteur de la facture principale du
                    # groupe, comme dans features.py.
                    primary_debtor_id = invoice_debtor.get(primary_invoice)
                    bankroll_code = lookups["debtor_bankroll_code"].get(primary_debtor_id)
                    seg_row = pd.Series(
                        dict(
                            market=market,
                            bankroll_code=bankroll_code,
                            amount_tier=_amount_tier(primary_payment_amount, amount_breakpoints),
                        )
                    )
                    tau_high = thresholds.global_tau_high
                    if thresholds.segment_columns:
                        key = tuple(seg_row.get(c) for c in thresholds.segment_columns)
                        tau_high = thresholds.per_segment_tau_high.get(key, thresholds.global_tau_high)

                    if proposal.mean_score >= tau_high and margin >= margin_delta:
                        decision = "AUTO_VALIDATION"
                    elif proposal.mean_score >= tau_low:
                        decision = "REVIEW"
                    else:
                        decision = "REJET"

                    decisions.append(DecidedGroup(current_day, proposal, decision, proposal.mean_score, margin))
                    if decision == "AUTO_VALIDATION":
                        for pid in proposal.payment_ids:
                            pending.pop(pid, None)

        expired = [pid for pid, entry in pending.items() if (current_day - entry).days > retention_days]
        for pid in expired:
            pending.pop(pid, None)

        current_day = next_day

    return decisions, pending
