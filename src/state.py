"""LedgerState : état du grand livre reconstruit au fil du journal (§8.2).

Toutes les méthodes de lecture prennent un `as_of` explicite. C'est la
garantie mécanique d'absence de fuite décrite en §8.2 : l'état ne peut
répondre que pour l'instant qu'il a atteint via `apply()`, jamais pour un
instant antérieur (déjà dépassé) ni en anticipant sur des événements futurs.

Convention retenue pour `current_amount` (cf. discussion §5.2 — le champ
`imputation.imputed_amount` utilisé par la formule de la spec n'existe pas
dans le schéma §2.3) : `residual_amount` est interprété comme le solde de la
facture immédiatement après la ligne d'imputation. `status = FULL` solde
toujours la facture à 0 ; `residual_amount` ne porte alors que l'écart
métier accepté (escompte, frais, retenue), pas un reste à payer réel.
"""

from __future__ import annotations

import dataclasses
from collections import defaultdict, deque
from typing import Optional

import pandas as pd

from src.events import Event
from src.normalize import normalize_label, normalize_reference


@dataclasses.dataclass
class _Invoice:
    invoice_id: str
    debtor_id: str
    agreement_id: str
    client_reference: str
    creation_date: pd.Timestamp
    due_date: pd.Timestamp
    currency: str
    initial_amount: int
    current_amount: int
    agreement_active_at_creation: bool
    last_status: Optional[str] = None
    reference_variants: frozenset[str] = dataclasses.field(default_factory=frozenset)


@dataclasses.dataclass
class _Party:
    party_id: str
    opened: bool = False
    closed: bool = False


@dataclasses.dataclass
class _Agreement:
    agreement_id: str
    debtor_id: str
    client_id: str
    market: str
    product: str
    recourse: bool
    active: bool = False
    disabled: bool = False


@dataclasses.dataclass
class _BehaviorObservation:
    timestamp: pd.Timestamp
    delay_days: int
    is_partial: bool
    is_grouped: bool
    cited_ref: bool


def _label_cites_reference(label: str, reference_variants: frozenset[str]) -> bool:
    if not label or not reference_variants:
        return False
    return bool(normalize_label(label).label_numbers & reference_variants)


class LedgerState:
    """État incrémental : `apply()` avance, les méthodes `*_at` lisent."""

    def __init__(self) -> None:
        self._clock: Optional[pd.Timestamp] = None
        self._invoices: dict[str, _Invoice] = {}
        self._parties: dict[str, _Party] = {}
        self._agreements: dict[str, _Agreement] = {}
        self._payments: dict[str, dict] = {}
        self._open_invoices_by_debtor: dict[str, set[str]] = defaultdict(set)
        self._behavior: dict[str, deque[_BehaviorObservation]] = defaultdict(deque)
        self._imputation_count_by_payment: dict[str, int] = defaultdict(int)
        self._imputation_count_by_invoice: dict[str, int] = defaultdict(int)
        # Index globaux pour l'étage A (blocking, §4) : ne contiennent que les
        # factures actuellement ouvertes (current_amount > 0), maintenus au
        # même rythme que le reste de l'état, jamais recalculés à la volée.
        self._reference_index: dict[str, set[str]] = defaultdict(set)
        self._amount_index: dict[int, set[str]] = defaultdict(set)
        # Historique de délai par couple (débiteur, agreement), non fenêtré
        # (cf. §5.3 `days_to_due_zscore`) — alimenté aux mêmes événements que
        # `_behavior`, juste indexé plus finement.
        self._delay_by_debtor_agreement: dict[tuple[str, str], list[float]] = defaultdict(list)

    # ---- avancement ---------------------------------------------------

    def apply(self, event: Event) -> None:
        """Avance l'état d'un événement. Le journal doit être trié (§8.1)."""
        if self._clock is not None and event.timestamp < self._clock:
            raise ValueError(
                f"journal non ordonné : {event.type} à {event.timestamp} "
                f"antérieur à l'horloge {self._clock}"
            )
        self._clock = event.timestamp
        handler = getattr(self, f"_apply_{event.type}", None)
        if handler is None:
            raise ValueError(f"type d'événement inconnu : {event.type}")
        handler(event.data)

    @property
    def clock(self) -> Optional[pd.Timestamp]:
        """Timestamp du dernier événement appliqué (None si aucun)."""
        return self._clock

    def _check_as_of(self, as_of: pd.Timestamp) -> None:
        if self._clock is not None and as_of < self._clock:
            raise ValueError(
                f"as_of={as_of} antérieur à l'horloge de l'état ({self._clock}) : "
                "impossible d'interroger le passé une fois l'état avancé."
            )

    # ---- handlers -------------------------------------------------------

    def _apply_PARTY_OPENED(self, data: dict) -> None:
        party = self._parties.setdefault(data["party_id"], _Party(data["party_id"]))
        party.opened = True

    def _apply_PARTY_CLOSED(self, data: dict) -> None:
        party = self._parties.setdefault(data["party_id"], _Party(data["party_id"]))
        party.closed = True

    def _apply_AGREEMENT_CREATED(self, data: dict) -> None:
        self._agreements[data["agreement_id"]] = _Agreement(
            agreement_id=data["agreement_id"],
            debtor_id=data["debtor_id"],
            client_id=data["client_id"],
            market=data["market"],
            product=data["product"],
            recourse=bool(data["recourse"]),
            active=True,
        )

    def _apply_AGREEMENT_DISABLED(self, data: dict) -> None:
        agreement = self._agreements.get(data["agreement_id"])
        if agreement is not None:
            agreement.disabled = True
            agreement.active = False

    def _apply_INVOICE_CREATED(self, data: dict) -> None:
        agreement = self._agreements.get(data["agreement_id"])
        invoice = _Invoice(
            invoice_id=data["invoice_id"],
            debtor_id=data["debtor_id"],
            agreement_id=data["agreement_id"],
            client_reference=data["client_reference"],
            creation_date=pd.Timestamp(data["creation_date"]),
            due_date=pd.Timestamp(data["due_date"]),
            currency=data["currency"],
            initial_amount=int(data["initial_amount"]),
            current_amount=int(data["initial_amount"]),
            agreement_active_at_creation=agreement is not None and agreement.active,
            reference_variants=normalize_reference(data["client_reference"]),
        )
        self._invoices[invoice.invoice_id] = invoice
        self._open_invoices_by_debtor[invoice.debtor_id].add(invoice.invoice_id)
        for variant in invoice.reference_variants:
            self._reference_index[variant].add(invoice.invoice_id)
        self._amount_index[invoice.current_amount].add(invoice.invoice_id)

    def _apply_PAYMENT_RECEIVED(self, data: dict) -> None:
        self._payments[data["payment_id"]] = data

    def _apply_IMPUTATION_APPLIED(self, data: dict) -> None:
        invoice = self._invoices.get(data["invoice_id"])
        payment = self._payments.get(data["payment_id"])
        if invoice is None:
            raise KeyError(f"imputation sur une facture inconnue de l'état : {data['invoice_id']}")
        if payment is None:
            raise KeyError(f"imputation sur un paiement inconnu de l'état : {data['payment_id']}")

        status = data["status"]
        residual = int(data["residual_amount"])
        old_amount = invoice.current_amount
        new_amount = 0 if status == "FULL" else residual
        invoice.current_amount = new_amount
        invoice.last_status = status

        if new_amount != old_amount:
            self._amount_index[old_amount].discard(invoice.invoice_id)
            if new_amount > 0:
                self._amount_index[new_amount].add(invoice.invoice_id)

        if new_amount <= 0:
            self._open_invoices_by_debtor[invoice.debtor_id].discard(invoice.invoice_id)
            for variant in invoice.reference_variants:
                self._reference_index[variant].discard(invoice.invoice_id)
        else:
            self._open_invoices_by_debtor[invoice.debtor_id].add(invoice.invoice_id)

        self._imputation_count_by_payment[payment["payment_id"]] += 1
        self._imputation_count_by_invoice[invoice.invoice_id] += 1
        is_grouped = (
            self._imputation_count_by_payment[payment["payment_id"]] > 1
            or self._imputation_count_by_invoice[invoice.invoice_id] > 1
        )

        delay_days = (pd.Timestamp(payment["value_date"]) - invoice.due_date).days
        cited_ref = _label_cites_reference(payment.get("label", ""), invoice.reference_variants)
        self._behavior[invoice.debtor_id].append(
            _BehaviorObservation(
                timestamp=pd.Timestamp(data["updated_at"]),
                delay_days=delay_days,
                is_partial=(status == "PARTIAL"),
                is_grouped=is_grouped,
                cited_ref=cited_ref,
            )
        )
        self._delay_by_debtor_agreement[(invoice.debtor_id, invoice.agreement_id)].append(
            float(delay_days)
        )

    # ---- lecture --------------------------------------------------------

    def open_invoices(self, debtor_id: str, as_of: pd.Timestamp) -> list[dict]:
        """Factures ouvertes (current_amount > 0) du débiteur, à `as_of`."""
        self._check_as_of(as_of)
        ids = self._open_invoices_by_debtor.get(debtor_id, set())
        return [dataclasses.asdict(self._invoices[i]) for i in ids]

    def invoices_by_reference(self, number_variants: frozenset[str], as_of: pd.Timestamp) -> list[dict]:
        """Factures ouvertes dont une variante de référence figure dans
        `number_variants` (typiquement `label_numbers` d'un libellé normalisé).
        Sans contrainte de débiteur ni de fenêtre temporelle (§4, clé K2)."""
        self._check_as_of(as_of)
        ids: set[str] = set()
        for variant in number_variants:
            ids |= self._reference_index.get(variant, set())
        return [dataclasses.asdict(self._invoices[i]) for i in ids]

    def invoices_by_amount(self, amount: int, as_of: pd.Timestamp) -> list[dict]:
        """Factures ouvertes dont le solde courant vaut exactement `amount`
        (§4, clé K3)."""
        self._check_as_of(as_of)
        ids = self._amount_index.get(amount, set())
        return [dataclasses.asdict(self._invoices[i]) for i in ids]

    def current_amount(self, invoice_id: str, as_of: pd.Timestamp) -> int:
        """Solde restant dû de la facture, tel que connu strictement avant `as_of`."""
        self._check_as_of(as_of)
        invoice = self._invoices.get(invoice_id)
        if invoice is None:
            raise KeyError(f"facture inconnue de l'état : {invoice_id}")
        return invoice.current_amount

    def get_invoice(self, invoice_id: str, as_of: pd.Timestamp) -> Optional[dict]:
        """Attributs connus de la facture à `as_of`, ou None si pas encore créée."""
        self._check_as_of(as_of)
        invoice = self._invoices.get(invoice_id)
        return None if invoice is None else dataclasses.asdict(invoice)

    def party_is_active(self, party_id: str, as_of: pd.Timestamp) -> bool:
        """Vrai si la partie (débiteur ou cédant) est ouverte et non fermée à `as_of`."""
        self._check_as_of(as_of)
        party = self._parties.get(party_id)
        return party is not None and party.opened and not party.closed

    def agreement_is_active(self, agreement_id: str, as_of: pd.Timestamp) -> bool:
        """Vrai si l'agreement a été créé et n'est pas désactivé à `as_of`."""
        self._check_as_of(as_of)
        agreement = self._agreements.get(agreement_id)
        return agreement is not None and agreement.active

    def get_agreement(self, agreement_id: str, as_of: pd.Timestamp) -> Optional[dict]:
        self._check_as_of(as_of)
        agreement = self._agreements.get(agreement_id)
        return None if agreement is None else dataclasses.asdict(agreement)

    def delay_stats(
        self, debtor_id: str, agreement_id: str, as_of: pd.Timestamp
    ) -> Optional[tuple[float, float, int]]:
        """(moyenne, écart-type, n) du délai de paiement du couple
        (débiteur, agreement), sur tout l'historique strictement antérieur à
        `as_of`. `None` si aucune observation (§5.3, `days_to_due_zscore`)."""
        self._check_as_of(as_of)
        values = self._delay_by_debtor_agreement.get((debtor_id, agreement_id), [])
        n = len(values)
        if n == 0:
            return None
        mean = sum(values) / n
        if n == 1:
            return (mean, 0.0, n)
        variance = sum((v - mean) ** 2 for v in values) / n
        return (mean, variance**0.5, n)

    def behavioral_stats(
        self, debtor_id: str, as_of: pd.Timestamp, window_days: int = 180
    ) -> dict:
        """Agrégats comportementaux du débiteur, fenêtre glissante strictement
        antérieure à `as_of` (§5.3, famille comportementale — consommée par
        `features.py` à partir de la Phase 6)."""
        self._check_as_of(as_of)
        window_start = as_of - pd.Timedelta(days=window_days)
        obs_deque = self._behavior[debtor_id]
        while obs_deque and obs_deque[0].timestamp < window_start:
            obs_deque.popleft()
        window_obs = [o for o in obs_deque if o.timestamp < as_of]

        open_ids = self._open_invoices_by_debtor.get(debtor_id, set())
        open_amount = sum(self._invoices[i].current_amount for i in open_ids)

        n = len(window_obs)
        if n == 0:
            return dict(
                debtor_mean_payment_delay=None,
                debtor_std_payment_delay=None,
                debtor_partial_payment_rate=None,
                debtor_grouping_rate=None,
                debtor_open_invoice_count=len(open_ids),
                debtor_open_invoice_amount=open_amount,
                debtor_ref_citation_rate=None,
                debtor_payment_count=0,
            )

        delays = pd.Series([o.delay_days for o in window_obs], dtype=float)
        return dict(
            debtor_mean_payment_delay=float(delays.mean()),
            debtor_std_payment_delay=float(delays.std(ddof=0)),
            debtor_partial_payment_rate=sum(o.is_partial for o in window_obs) / n,
            debtor_grouping_rate=sum(o.is_grouped for o in window_obs) / n,
            debtor_open_invoice_count=len(open_ids),
            debtor_open_invoice_amount=open_amount,
            debtor_ref_citation_rate=sum(o.cited_ref for o in window_obs) / n,
            debtor_payment_count=n,
        )
