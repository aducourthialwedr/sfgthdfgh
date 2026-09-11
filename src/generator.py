"""Générateur de données synthétiques pour le moteur de rapprochement.

Écrit les six tables du modèle de données (§2 de la spec) plus un fichier
`ground_truth.parquet` qui sert de vérité terrain pour mesurer les
performances des étages ultérieurs.

Toutes les grandeurs monétaires sont des entiers de centimes (jamais de
float). Le générateur est déterministe : même `seed`, mêmes données.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import time
from typing import Optional

import numpy as np
import pandas as pd


def _log(msg: str) -> None:
    """Progression optionnelle (`verbose=True`) — silencieux par défaut,
    utilisé par les tests et le pipeline normal sans polluer leur sortie."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------
# Enums (choix arbitraires du générateur, non imposés par la spec)
# --------------------------------------------------------------------------

CHANNELS = ["SEPA", "SWIFT", "CHEQUE", "LCR"]
PAYMENT_TYPES = ["VIREMENT", "PRELEVEMENT", "REMISE_CHEQUE", "LCR"]
BANKROLL_CODES = ["STANDARD", "CONFIDENTIEL", "SOUS_PARTICIPATION"]
MARKETS = ["BTP", "INDUSTRIE", "SERVICES", "COMMERCE"]
_MARKET_PROBS = [0.25, 0.30, 0.30, 0.15]  # doit rester aligné avec _gen_parties
PRODUCTS = ["CLASSIQUE", "INVERSE", "CONFIDENTIEL"]
PAYER_TYPES = ["PONCTUEL", "RETARDATAIRE_CHRONIQUE", "GROUPE"]

# Délai moyen de paiement (jours après due_date) par profil de débiteur.
_PAYER_DELAY_MEAN = {"PONCTUEL": 2.0, "RETARDATAIRE_CHRONIQUE": 25.0, "GROUPE": 8.0}
_PAYER_DELAY_STD = {"PONCTUEL": 3.0, "RETARDATAIRE_CHRONIQUE": 12.0, "GROUPE": 5.0}
_MARKET_DELAY_OFFSET = {"BTP": 8.0, "INDUSTRIE": 0.0, "SERVICES": -2.0, "COMMERCE": 0.0}
_MARKET_PAYMENT_TERM = {"BTP": 55, "INDUSTRIE": 40, "SERVICES": 30, "COMMERCE": 35}

CASE_TYPES = [
    "1-1_clean",
    "1-1_noisy",
    "1-n",
    "n-1",
    "n-n",
    "discount",
    "bank_fee",
    "retention_btp",
    "no_invoice",
]

DEFAULT_CASE_WEIGHTS: dict[str, float] = {
    "1-1_clean": 0.55,
    "1-1_noisy": 0.15,
    "1-n": 0.12,
    "n-1": 0.08,
    "n-n": 0.03,
    "discount": 0.03,
    "bank_fee": 0.02,
    "retention_btp": 0.01,
    "no_invoice": 0.01,
}

# Pondération alternative approchant le ratio factures/paiements observé en
# production (~1.7M factures / 500k paiements ≈ 3,4:1) — bien plus de
# regroupement 1↔n que le mix par défaut. Utilisée pour les benchmarks de
# performance à volume réel, pas pour l'entraînement (qui n'a pas besoin de
# coller à ce ratio précis).
HIGH_GROUPING_CASE_WEIGHTS: dict[str, float] = {
    "1-1_clean": 0.08,
    "1-1_noisy": 0.04,
    "1-n": 0.72,
    "n-1": 0.08,
    "n-n": 0.04,
    "discount": 0.02,
    "bank_fee": 0.01,
    "retention_btp": 0.005,
    "no_invoice": 0.005,
}

_SURNAMES = [
    "MARTIN", "BERNARD", "DUBOIS", "THOMAS", "ROBERT", "PETIT", "DURAND", "LEROY",
    "MOREAU", "SIMON", "LAURENT", "LEFEBVRE", "MICHEL", "GARCIA", "DAVID", "BERTRAND",
    "ROUX", "VINCENT", "FONTAINE", "CHEVALIER", "GAUTHIER", "MASSON", "DUPONT",
    "LAMBERT", "BONNET", "FRANCOIS", "MARTINEZ", "LEGRAND", "GARNIER", "FAURE",
    "ANDRE", "MERCIER", "BLANC", "GUERIN", "BOYER", "CLEMENT", "MEYER", "GAUTIER",
    "GIRARD", "ROBIN", "MOLINA", "MULLER", "LEROUX", "COLIN", "NOEL", "PERRIN",
    "MOREL", "RENARD", "GIRAUD", "BOURGEOIS", "DENIS", "DUMONT", "MARCHAND",
    "DUFOUR", "DUMAS", "MARIE", "BRUN", "PICARD", "MEUNIER", "SCHMITT", "ROY",
    "JOLY", "CARON", "GAILLARD", "ROUSSEAU", "OLIVIER", "PIERRE", "BOULANGER",
    "LOUIS", "LEBRUN", "ARNAUD", "GUILLOT", "GUILLAUME", "CHARLES", "HUBERT",
    "ROLLAND", "JEAN", "MASSE", "BOUCHER", "PAUL", "HUET", "LUCAS", "FERNANDEZ",
    "RENAUD", "ADAM", "PONS", "PERRET", "MOULIN", "FISCHER", "PASQUIER", "GUYOT",
    "BENOIT", "VIDAL", "PICOT", "COLAS", "VASSEUR", "REY", "DELAUNAY", "BRIAND",
    "BOUVIER", "LEMAIRE", "LEMOINE", "COUSIN", "LACROIX", "MAILLARD", "TESSIER",
    "GERARD", "SEGUIN", "LEGER", "COLLET", "LOPEZ", "HENRY", "RIVIERE", "PELLETIER",
    "COUTURIER", "CORDIER", "MALLET", "VILLA", "BARBIER", "SANCHEZ", "AUBRY",
    "AUBERT", "REYNAUD", "HERVE", "SALMON", "PRUDHOMME", "FOURNIER", "CHARPENTIER",
    "GAY", "MICHAUD", "VERNET", "BAILLY", "LEFORT", "BAILLEUL", "PAGE", "JULIEN",
    "GIROUX", "SIMONET", "DAGUENET", "PELTIER", "TISSOT", "PERNOT", "MARCHAL",
    "GENTIL", "VALLET", "COQUET", "SALLE", "BODIN", "BRETON", "CAILLE",
    "CHAUVIN", "COUSTEAU", "DEVAUX", "ETIENNE", "FABRE", "GALLET", "HAMON",
    "ISAMBERT", "JANIN", "KELLER", "LACOMBE", "MAGNIN", "NAVARRE", "ORY", "OZENNE",
    "QUENTIN", "RAMBAUD", "SAUVAGE", "TARDIF", "URVOY", "VALOIS", "WEBER", "PAULIN",
]

# Complète _SURNAMES par combinaison préfixe+suffixe (~48*48 = 2304
# combinaisons) : la liste ci-dessus (173 entrées) est lisible pour des
# démonstrations, mais bien trop peu diversifiée pour rester réaliste à
# plusieurs millions de débiteurs — un token de nom serait alors partagé
# par des milliers de débiteurs, et l'élagage dynamique des tokens trop
# fréquents dans l'index K4 (§4, `blocking.build_static_lookups`) finirait
# par tout éliminer, désactivant K4 de fait. Un vrai référentiel de noms a
# une cardinalité bien plus grande ; cette combinatoire l'approche sans
# maintenir des dizaines de milliers de noms à la main.
_SURNAME_PREFIXES = [
    "MAR", "BER", "DUB", "THO", "ROB", "PET", "DUR", "LER", "MOR", "SIM",
    "LAU", "LEF", "MIC", "GAR", "DAV", "BOU", "ROU", "VIN", "FON", "CHE",
    "GAU", "MAS", "DUP", "LAM", "BON", "FRA", "LEG", "FAU", "AND", "MER",
    "BLA", "GUE", "BOY", "CLE", "MEY", "GIR", "MOL", "MUL", "COL", "NOE",
    "PER", "REN", "DEN", "DUM", "MARC", "DUF", "BRU", "PIC",
]
_SURNAME_SUFFIXES = [
    "TIN", "NARD", "OIS", "MAS", "ERT", "IER", "URE", "AULT", "ANT", "ILLE",
    "EAU", "OT", "AND", "ARD", "ETTE", "OU", "AUD", "ON", "ET", "IN",
    "OY", "EY", "ARDIN", "EUIL", "IGNON", "UZE", "ONS", "AGNE", "ERIE", "ISSE",
    "OUX", "UET", "INEAU", "ELLE", "ANCE", "ENSE", "ORT", "USSE", "IVET", "ONCE",
    "ARY", "ENAY", "INOT", "OUARD", "ASSE", "ONDE", "URY", "ELIN",
]
_ALL_SURNAMES: np.ndarray = np.array(
    list(_SURNAMES) + [f"{p}{s}" for p in _SURNAME_PREFIXES for s in _SURNAME_SUFFIXES]
)
# `rng.choice` sur une liste Python reconvertit en tableau à *chaque appel* :
# coût caché négligeable une fois, mais rédhibitoire appelé ~n_debtors fois
# (même piège que les autres correctifs de cette passe). Un tableau numpy
# précalculé une seule fois évite cette reconversion répétée.

_LEGAL_FORMS = ["SARL", "SAS", "SA", "EURL", "SNC", "SASU"]
_SECTOR_WORDS = {
    "BTP": ["BATIMENT", "CONSTRUCTION", "TP", "RENOVATION"],
    "INDUSTRIE": ["INDUSTRIE", "MECANIQUE", "USINAGE", "METALLURGIE"],
    "SERVICES": ["SERVICES", "CONSEIL", "INGENIERIE"],
    "COMMERCE": ["DISTRIBUTION", "NEGOCE", "COMMERCE"],
}
_LABEL_PREFIXES = ["VIR", "VIREMENT", "PAIEMENT", "REGLEMENT", "VRT"]


@dataclasses.dataclass
class GeneratorParams:
    """Paramètres du générateur, reproductible via `seed`."""

    n_debtors: int = 200
    n_assignors: int = 30
    n_invoices: int = 20_000
    n_months: int = 14
    seed: int = 42
    start_date: dt.date = dt.date(2024, 1, 1)
    case_weights: dict[str, float] = dataclasses.field(
        default_factory=lambda: dict(DEFAULT_CASE_WEIGHTS)
    )

    @property
    def end_date(self) -> dt.date:
        return self.start_date + dt.timedelta(days=30 * self.n_months)


# --------------------------------------------------------------------------
# Helpers : identités, IBAN, noms
# --------------------------------------------------------------------------


def _iban_check_digits(bban: str, country: str = "FR") -> str:
    rearranged = bban + country + "00"
    numeric = "".join(str(int(ch, 36)) for ch in rearranged)
    remainder = int(numeric) % 97
    return f"{98 - remainder:02d}"


def _fake_iban(rng: np.random.Generator) -> str:
    bban = "".join(str(rng.integers(0, 10)) for _ in range(23))
    check = _iban_check_digits(bban)
    return f"FR{check}{bban}"


def _fake_company_name(rng: np.random.Generator, market: Optional[str] = None) -> str:
    surname = str(rng.choice(_ALL_SURNAMES))
    form = rng.choice(_LEGAL_FORMS)
    if market is not None and rng.random() < 0.6:
        word = rng.choice(_SECTOR_WORDS[market])
        return f"{form} {surname} {word}"
    return f"{form} {surname}"


# --------------------------------------------------------------------------
# Helpers : bruit de libellé
# --------------------------------------------------------------------------


def _mangle_reference(rng: np.random.Generator, ref: str) -> str:
    s = ref
    if rng.random() < 0.3:
        s = "".join(ch for ch in s if not ch.isalpha())
    if rng.random() < 0.3 and s[:1] == "0":
        s = s.lstrip("0") or "0"
    if rng.random() < 0.2:
        s = "0" * int(rng.integers(1, 4)) + s
    if rng.random() < 0.2 and len(s) > 5:
        cut = int(rng.integers(5, len(s) + 1))
        s = s[:cut]
    if rng.random() < 0.15:
        prefix = rng.choice(["REF", "FACT", "N", "INV"])
        s = f"{prefix}{s}"
    return s


def _mangle_name(rng: np.random.Generator, name: str) -> str:
    tokens = name.split()
    if rng.random() < 0.3 and len(tokens) > 1:
        tokens = tokens[1:]
    if rng.random() < 0.2:
        tokens = [t[:4] for t in tokens]
    if rng.random() < 0.15 and tokens:
        idx = int(rng.integers(0, len(tokens)))
        chars = list(tokens[idx])
        if len(chars) > 3:
            i = int(rng.integers(0, len(chars) - 1))
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
        tokens[idx] = "".join(chars)
    return " ".join(tokens)


_ACCENT_MAP = {"E": "É", "A": "À", "U": "Ü", "C": "Ç"}


def _randomize_accents_and_case(rng: np.random.Generator, s: str) -> str:
    chars = []
    for ch in s:
        if ch in _ACCENT_MAP and rng.random() < 0.05:
            ch = _ACCENT_MAP[ch]
        chars.append(ch)
    s = "".join(chars)
    r = rng.random()
    if r < 0.4:
        return s.lower()
    if r < 0.6:
        return s.title()
    return s


def build_label(
    rng: np.random.Generator,
    ref: str,
    debtor_name: str,
    include_ref: bool,
    noise_level: str,
) -> str:
    """Construit un libellé bancaire bruité.

    `noise_level` ∈ {"low", "high", "empty", "numeric_only"}.
    """
    if noise_level == "empty":
        return ""
    if noise_level == "numeric_only":
        digits = "".join(ch for ch in ref if ch.isdigit())
        if rng.random() < 0.5:
            digits = _mangle_reference(rng, digits)
        return digits

    parts = []
    if rng.random() < 0.7:
        parts.append(str(rng.choice(_LABEL_PREFIXES)))
    name_part = _mangle_name(rng, debtor_name) if noise_level == "high" else debtor_name
    if rng.random() < 0.85:
        parts.append(name_part)
    if include_ref:
        ref_part = _mangle_reference(rng, ref) if noise_level == "high" else ref
        parts.append(ref_part)
    label = " ".join(parts).strip()
    if noise_level == "high":
        label = _randomize_accents_and_case(rng, label)
    return label


# --------------------------------------------------------------------------
# Génération des parties (débiteurs, cédants) et contrats
# --------------------------------------------------------------------------


def _gen_parties(
    rng: np.random.Generator,
    n: int,
    prefix: str,
    params: GeneratorParams,
    is_debtor: bool,
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Génère `debtor` ou `assignor`. Schéma réel observé : `debtor` n'a pas
    de `closed_at` (un débiteur n'est jamais formellement fermé) ; seul
    `assignor` en porte un. `opened_at` existe pour les deux."""
    rows = []
    profiles = []
    bankroll = rng.choice(BANKROLL_CODES, size=n, p=[0.7, 0.2, 0.1])
    dominant_market = rng.choice(MARKETS, size=n, p=_MARKET_PROBS)
    payer_type = rng.choice(PAYER_TYPES, size=n, p=[0.5, 0.25, 0.25])
    ref_citation_rate = rng.beta(2, 2, size=n)
    log_every = max(1, n // 20)
    for i in range(n):
        if verbose and i > 0 and i % log_every == 0:
            _log(f"  {prefix} : {i:,}/{n:,}")
        party_id = f"{prefix}{i + 1:05d}"
        market = dominant_market[i]
        name = _fake_company_name(rng, market=market if is_debtor else None)
        iban = _fake_iban(rng)
        opened_at = params.start_date - dt.timedelta(days=int(rng.integers(30, 1000)))
        row = dict(
            party_id=party_id,
            bankroll_code=bankroll[i],
            iban=iban,
            name=name,
            opened_at=pd.Timestamp(opened_at),
        )
        if not is_debtor:
            closed_at = pd.NaT
            if rng.random() < 0.05:
                span = (params.end_date - params.start_date).days
                closed_at = pd.Timestamp(params.start_date) + pd.Timedelta(
                    days=int(rng.integers(60, max(61, span)))
                )
            row["closed_at"] = closed_at
        rows.append(row)
        if is_debtor:
            profiles.append(
                dict(
                    party_id=party_id,
                    payer_type=payer_type[i],
                    ref_citation_rate=float(ref_citation_rate[i]),
                    dominant_market=market,
                )
            )
    parties_df = pd.DataFrame(rows)
    profiles_df = pd.DataFrame(profiles) if is_debtor else pd.DataFrame()
    return parties_df, profiles_df


def _gen_agreements(
    rng: np.random.Generator,
    debtors: pd.DataFrame,
    debtor_profiles: pd.DataFrame,
    assignors: pd.DataFrame,
    params: GeneratorParams,
    verbose: bool = False,
) -> pd.DataFrame:
    rows = []
    agr_seq = 0
    # dict Python plutôt que Series.loc/iterrows (~10-100x plus rapide à
    # grande échelle : chaque .loc[]/iterrows() a un coût pandas fixe par
    # appel, dominant sur plusieurs millions d'itérations).
    market_by_debtor = dict(zip(debtor_profiles["party_id"], debtor_profiles["dominant_market"]))
    assignor_ids = assignors["party_id"].to_numpy()
    debtor_ids = debtors["party_id"].to_numpy()
    n_debtors = len(debtor_ids)
    log_every = max(1, n_debtors // 20)

    for i, debtor_id in enumerate(debtor_ids):
        if verbose and i > 0 and i % log_every == 0:
            _log(f"  agreements : {i:,}/{n_debtors:,} débiteurs traités ({len(rows):,} agreements)")
        market = market_by_debtor[debtor_id]
        n_agreements = int(rng.integers(1, 4))
        chosen_assignors = rng.choice(assignor_ids, size=n_agreements, replace=False)
        for j, assignor_id in enumerate(chosen_assignors):
            agr_seq += 1
            agreement_id = f"AGR{agr_seq:06d}"
            created_at = params.start_date - dt.timedelta(days=int(rng.integers(0, 500)))
            disabled_at = pd.NaT
            # Le premier contrat de chaque débiteur reste actif toute la période
            # (garantit qu'il existe toujours au moins un agreement utilisable).
            if j > 0 and rng.random() < 0.05:
                span = (params.end_date - params.start_date).days
                disabled_at = pd.Timestamp(params.start_date) + pd.Timedelta(
                    days=int(rng.integers(60, max(61, span)))
                )
            rows.append(
                dict(
                    agreement_id=agreement_id,
                    debtor_id=debtor_id,
                    client_id=assignor_id,
                    contract_number=f"CTR{agr_seq:06d}",
                    created_at=pd.Timestamp(created_at),
                    disabled_at=disabled_at,
                    market=market,
                    product=str(rng.choice(PRODUCTS)),
                    recourse=bool(rng.random() < 0.6),
                )
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Génération des factures (attributs de base, sans montant restant final)
# --------------------------------------------------------------------------


def _gen_invoices_base(
    rng: np.random.Generator,
    debtors: pd.DataFrame,
    agreements: pd.DataFrame,
    params: GeneratorParams,
    verbose: bool = False,
) -> pd.DataFrame:
    rows = []
    # Listes de dicts Python plutôt que DataFrame filtré + .sample() par
    # facture : `.sample(random_state=...)` recrée un objet RandomState à
    # chaque appel (coût pandas connu, dominant sur des millions d'appels),
    # et le filtrage booléen a lui aussi un coût pandas fixe par appel.
    agreements_by_debtor: dict[str, list[dict]] = {
        d: g.to_dict("records") for d, g in agreements.groupby("debtor_id")
    }
    debtor_ids = debtors["party_id"].to_numpy()
    span_days = (params.end_date - params.start_date).days
    log_every = max(1, params.n_invoices // 20)

    seq = 0
    for idx in range(params.n_invoices):
        if verbose and idx > 0 and idx % log_every == 0:
            _log(f"  factures : {idx:,}/{params.n_invoices:,}")
        debtor_id = rng.choice(debtor_ids)
        # Un débiteur n'a pas de date de fermeture (schéma réel) : seule la
        # fenêtre globale de génération borne la date de création.
        creation_offset = int(rng.integers(0, max(1, span_days)))
        creation_date = params.start_date + dt.timedelta(days=creation_offset)

        candidate_agreements = agreements_by_debtor.get(debtor_id)
        if not candidate_agreements:
            continue
        creation_ts = pd.Timestamp(creation_date)
        active = [
            a
            for a in candidate_agreements
            if a["created_at"] <= creation_ts
            and (pd.isna(a["disabled_at"]) or a["disabled_at"] > creation_ts)
        ]
        if not active:
            continue
        agreement = active[int(rng.integers(0, len(active)))]

        seq += 1
        invoice_id = f"INV{seq:06d}"
        term = _MARKET_PAYMENT_TERM[agreement["market"]]
        due_date = creation_date + dt.timedelta(days=term)
        amount_eur = float(np.clip(rng.lognormal(mean=8.3, sigma=0.7), 80, 60_000))
        initial_amount = int(round(amount_eur * 100))
        client_reference = f"FA{creation_date:%y}{seq:06d}"

        rows.append(
            dict(
                invoice_id=invoice_id,
                client_reference=client_reference,
                creation_date=pd.Timestamp(creation_date),
                due_date=pd.Timestamp(due_date),
                initial_amount=initial_amount,
                currency="EUR",
                debtor_id=debtor_id,
                agreement_id=agreement["agreement_id"],
                market=agreement["market"],
            )
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Plan de règlement : assigne un case_type à chaque facture / groupe
# --------------------------------------------------------------------------


def _build_settlement_plans(
    rng: np.random.Generator, invoices: pd.DataFrame, case_weights: dict[str, float]
) -> list[dict]:
    """Regroupe les factures de chaque débiteur en plans de règlement."""
    invoice_case_types = [c for c in CASE_TYPES if c != "no_invoice"]
    base_weights = np.array([case_weights[c] for c in invoice_case_types], dtype=float)
    retention_idx = invoice_case_types.index("retention_btp")
    # La retenue de garantie n'existe qu'en BTP (~25 % des débiteurs) : on
    # compense en boostant son poids conditionnel pour retomber sur la
    # proportion cible une fois moyenné sur toute la population.
    btp_boost = 1.0 / _MARKET_PROBS[MARKETS.index("BTP")]

    plans = []
    for debtor_id, group in invoices.groupby("debtor_id", sort=False):
        pending = group.sort_values("creation_date").to_dict("records")
        while pending:
            n_pending = len(pending)
            is_btp = pending[0]["market"] == "BTP"
            feasible = np.array(
                [
                    True,  # 1-1_clean
                    True,  # 1-1_noisy
                    n_pending >= 2,  # 1-n
                    True,  # n-1
                    n_pending >= 2,  # n-n
                    True,  # discount
                    True,  # bank_fee
                    is_btp,  # retention_btp
                ],
                dtype=bool,
            )
            w = base_weights * feasible
            if is_btp:
                w[retention_idx] *= btp_boost
            if w.sum() <= 0:
                w = feasible.astype(float)
            w = w / w.sum()
            case_type = rng.choice(invoice_case_types, p=w)

            if case_type == "1-n":
                k = int(rng.integers(2, min(5, n_pending) + 1))
                invs = [pending.pop(0) for _ in range(k)]
            elif case_type == "n-n":
                k = int(rng.integers(2, min(4, n_pending) + 1))
                invs = [pending.pop(0) for _ in range(k)]
            else:
                invs = [pending.pop(0)]

            plans.append(dict(debtor_id=debtor_id, case_type=case_type, invoices=invs))
    return plans


# --------------------------------------------------------------------------
# Matérialisation des paiements / imputations à partir des plans
# --------------------------------------------------------------------------


def _sample_delay_days(rng: np.random.Generator, payer_type: str, market: str) -> int:
    mean = _PAYER_DELAY_MEAN[payer_type] + _MARKET_DELAY_OFFSET[market]
    std = _PAYER_DELAY_STD[payer_type]
    delay = rng.normal(mean, std)
    return int(round(np.clip(delay, -10, 120)))


def _pick_iban_route(
    rng: np.random.Generator, debtor_iban: str, assignor_iban: str, technical_ibans: list[str]
) -> str:
    """Retourne l'IBAN émetteur du paiement. `payment` n'a pas de
    `bankroll_code` propre (schéma réel) : le compte technique se
    reconnaît par appartenance à un pool d'IBAN techniques connus, pas par
    un champ dédié — voir `blocking.resolve_iban`."""
    r = rng.random()
    if r < 0.85:
        return debtor_iban  # DEBTOR_DIRECT
    if r < 0.93:
        return assignor_iban  # ASSIGNOR
    return str(rng.choice(technical_ibans))  # TECHNICAL_ACCOUNT


@dataclasses.dataclass
class _Counters:
    payment_seq: int = 0
    group_seq: int = 0

    def next_payment_id(self) -> str:
        self.payment_seq += 1
        return f"PMT{self.payment_seq:07d}"

    def next_group_id(self) -> str:
        self.group_seq += 1
        return f"GRP{self.group_seq:06d}"


def _materialize_plan(
    rng: np.random.Generator,
    plan: dict,
    debtor_row: pd.Series,
    debtor_profile: pd.Series,
    assignor_iban: str,
    collector_ibans: list[str],
    technical_ibans: list[str],
    counters: _Counters,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Construit (payment_rows, imputation_rows, ground_truth_rows, invoice_final_rows)."""
    case_type = plan["case_type"]
    invs = plan["invoices"]
    payer_type = debtor_profile["payer_type"]
    cites_ref = rng.random() < debtor_profile["ref_citation_rate"]
    group_id = counters.next_group_id()

    payments: list[dict] = []
    imputations: list[dict] = []
    gts: list[dict] = []
    finals: list[dict] = []

    # `updated_at` doit être strictement croissant dans l'ordre logique de
    # règlement (l'ordre dans lequel les lignes d'imputation sont ajoutées
    # ci-dessous) : sinon le rejeu événementiel appliquerait une ligne
    # PARTIAL après la ligne FULL censée clôturer la facture, et l'état
    # reconstruit divergerait du solde final voulu par le générateur.
    _last_updated_at: list[pd.Timestamp] = [pd.Timestamp.min]

    def next_updated_at(candidate: pd.Timestamp) -> pd.Timestamp:
        ts = max(pd.Timestamp(candidate), _last_updated_at[0] + pd.Timedelta(seconds=1))
        _last_updated_at[0] = ts
        return ts

    def make_payment(value_date: dt.date, amount: int, ref_for_label: str, noisy: bool) -> dict:
        iban_debtor = _pick_iban_route(rng, debtor_row["iban"], assignor_iban, technical_ibans)
        include_ref = cites_ref and not noisy
        if noisy:
            noise_level = str(rng.choice(["high", "empty", "numeric_only"], p=[0.5, 0.3, 0.2]))
        else:
            noise_level = "low"
        label = build_label(rng, ref_for_label, debtor_row["name"], include_ref, noise_level)
        return dict(
            payment_id=counters.next_payment_id(),
            value_date=pd.Timestamp(value_date),
            amount=amount,
            currency="EUR",
            iban_debtor=iban_debtor,
            iban_creditor=str(rng.choice(collector_ibans)),
            label=label,
            channel=str(rng.choice(CHANNELS, p=[0.6, 0.1, 0.2, 0.1])),
            payment_type=str(rng.choice(PAYMENT_TYPES, p=[0.7, 0.1, 0.15, 0.05])),
        )

    if case_type in ("1-1_clean", "1-1_noisy", "discount", "bank_fee", "retention_btp"):
        inv = invs[0]
        delay = _sample_delay_days(rng, payer_type, inv["market"])
        value_date = inv["due_date"].date() + dt.timedelta(days=delay)
        noisy = case_type == "1-1_noisy"
        if case_type == "discount":
            rate = rng.uniform(0.005, 0.03)
            amount = int(round(inv["initial_amount"] * (1 - rate)))
        elif case_type == "bank_fee":
            fee = int(rng.integers(500, 4001))
            amount = inv["initial_amount"] - fee
        elif case_type == "retention_btp":
            rate = rng.uniform(0.04, 0.06)
            amount = int(round(inv["initial_amount"] * (1 - rate)))
        else:
            amount = inv["initial_amount"]
        residual = inv["initial_amount"] - amount

        pay = make_payment(value_date, amount, inv["client_reference"], noisy)
        payments.append(pay)
        updated_at = next_updated_at(
            pd.Timestamp(value_date) + pd.Timedelta(hours=int(rng.integers(1, 48)))
        )
        imputations.append(
            dict(
                payment_id=pay["payment_id"],
                invoice_id=inv["invoice_id"],
                status="FULL",
                updated_at=updated_at,
                residual_amount=residual,
            )
        )
        gts.append(
            dict(
                payment_id=pay["payment_id"],
                invoice_id=inv["invoice_id"],
                case_type=case_type,
                group_id=group_id,
                status="FULL",
                residual_amount=residual,
            )
        )
        finals.append(dict(invoice_id=inv["invoice_id"], current_amount=0))

    elif case_type == "1-n":
        total = sum(i["initial_amount"] for i in invs)
        ref_for_label = invs[0]["client_reference"]
        max_due = max(i["due_date"].date() for i in invs)
        delay = _sample_delay_days(rng, payer_type, invs[0]["market"])
        value_date = max_due + dt.timedelta(days=delay)
        pay = make_payment(value_date, total, ref_for_label, noisy=False)
        payments.append(pay)
        updated_at = next_updated_at(
            pd.Timestamp(value_date) + pd.Timedelta(hours=int(rng.integers(1, 48)))
        )
        for inv in invs:
            imputations.append(
                dict(
                    payment_id=pay["payment_id"],
                    invoice_id=inv["invoice_id"],
                    status="FULL",
                    updated_at=updated_at,
                    residual_amount=0,
                )
            )
            gts.append(
                dict(
                    payment_id=pay["payment_id"],
                    invoice_id=inv["invoice_id"],
                    case_type=case_type,
                    group_id=group_id,
                    status="FULL",
                    residual_amount=0,
                )
            )
            finals.append(dict(invoice_id=inv["invoice_id"], current_amount=0))

    elif case_type == "n-1":
        inv = invs[0]
        n_splits = int(rng.integers(2, 4))
        fractions = rng.dirichlet(np.ones(n_splits))
        remaining = inv["initial_amount"]
        base_delay = _sample_delay_days(rng, payer_type, inv["market"])
        for k in range(n_splits):
            is_last = k == n_splits - 1
            amount = remaining if is_last else int(round(inv["initial_amount"] * fractions[k]))
            amount = max(amount, 1)
            amount = min(amount, remaining)
            value_date = inv["due_date"].date() + dt.timedelta(days=base_delay + k * int(rng.integers(5, 30)))
            pay = make_payment(value_date, amount, inv["client_reference"], noisy=False)
            payments.append(pay)
            updated_at = next_updated_at(
                pd.Timestamp(value_date) + pd.Timedelta(hours=int(rng.integers(1, 48)))
            )
            remaining -= amount
            status = "FULL" if is_last else "PARTIAL"
            imputations.append(
                dict(
                    payment_id=pay["payment_id"],
                    invoice_id=inv["invoice_id"],
                    status=status,
                    updated_at=updated_at,
                    residual_amount=remaining,
                )
            )
            gts.append(
                dict(
                    payment_id=pay["payment_id"],
                    invoice_id=inv["invoice_id"],
                    case_type=case_type,
                    group_id=group_id,
                    status=status,
                    residual_amount=remaining,
                )
            )
        finals.append(dict(invoice_id=inv["invoice_id"], current_amount=max(remaining, 0)))

    elif case_type == "n-n":
        # Étalement en cascade ("waterfall") : n paiements dans une fenêtre de 72h
        # couvrent n factures, chaque ligne d'imputation reste une paire bien définie.
        n_payments = max(2, len(invs) - 1) if len(invs) > 2 else 2
        total = sum(i["initial_amount"] for i in invs)
        fractions = rng.dirichlet(np.ones(n_payments))
        pay_amounts = [max(1, int(round(total * f))) for f in fractions]
        pay_amounts[-1] += total - sum(pay_amounts)

        base_delay = _sample_delay_days(rng, payer_type, invs[0]["market"])
        base_date = pd.Timestamp(max(i["due_date"].date() for i in invs)) + pd.Timedelta(days=base_delay)
        window_hours = sorted(int(rng.integers(0, 72)) for _ in range(n_payments))

        inv_queue = list(invs)
        remaining_inv_amount = inv_queue[0]["initial_amount"] if inv_queue else 0
        for k, amount in enumerate(pay_amounts):
            value_date = base_date + pd.Timedelta(hours=window_hours[k])
            ref_for_label = inv_queue[0]["client_reference"] if inv_queue else invs[0]["client_reference"]
            pay = make_payment(value_date.date(), amount, ref_for_label, noisy=False)
            payments.append(pay)
            updated_at = next_updated_at(
                pd.Timestamp(value_date) + pd.Timedelta(hours=int(rng.integers(1, 24)))
            )
            to_allocate = amount
            while to_allocate > 0 and inv_queue:
                current_inv = inv_queue[0]
                take = min(to_allocate, remaining_inv_amount)
                to_allocate -= take
                remaining_inv_amount -= take
                status = "FULL" if remaining_inv_amount == 0 else "PARTIAL"
                imputations.append(
                    dict(
                        payment_id=pay["payment_id"],
                        invoice_id=current_inv["invoice_id"],
                        status=status,
                        updated_at=updated_at,
                        residual_amount=remaining_inv_amount,
                    )
                )
                gts.append(
                    dict(
                        payment_id=pay["payment_id"],
                        invoice_id=current_inv["invoice_id"],
                        case_type=case_type,
                        group_id=group_id,
                        status=status,
                        residual_amount=remaining_inv_amount,
                    )
                )
                if remaining_inv_amount == 0:
                    finals.append(dict(invoice_id=current_inv["invoice_id"], current_amount=0))
                    inv_queue.pop(0)
                    if inv_queue:
                        remaining_inv_amount = inv_queue[0]["initial_amount"]
        # Reliquat non couvert si les montants ne s'équilibrent pas exactement.
        for inv in inv_queue:
            finals.append(
                dict(invoice_id=inv["invoice_id"], current_amount=max(remaining_inv_amount, 0))
            )

    else:
        raise ValueError(f"case_type inattendu dans _materialize_plan: {case_type}")

    return payments, imputations, gts, finals


def _gen_orphan_payments(
    rng: np.random.Generator,
    n: int,
    debtors: pd.DataFrame,
    assignors: pd.DataFrame,
    collector_ibans: list[str],
    technical_ibans: list[str],
    params: GeneratorParams,
    counters: _Counters,
) -> list[dict]:
    rows = []
    span = (params.end_date - params.start_date).days
    debtor_records = debtors[["iban", "name"]].to_dict("records")
    assignor_ibans = assignors["iban"].to_numpy()
    for _ in range(n):
        debtor = debtor_records[int(rng.integers(0, len(debtor_records)))]
        value_date = params.start_date + dt.timedelta(days=int(rng.integers(0, span)))
        amount_eur = float(np.clip(rng.lognormal(mean=7.5, sigma=0.8), 20, 20_000))
        amount = int(round(amount_eur * 100))
        fake_ref = f"FA{value_date:%y}{int(rng.integers(0, 999999)):06d}"
        iban_debtor = _pick_iban_route(
            rng, debtor["iban"], str(rng.choice(assignor_ibans)), technical_ibans
        )
        label = build_label(
            rng, fake_ref, debtor["name"], include_ref=rng.random() < 0.3, noise_level="high"
        )
        rows.append(
            dict(
                payment_id=counters.next_payment_id(),
                value_date=pd.Timestamp(value_date),
                amount=amount,
                currency="EUR",
                iban_debtor=iban_debtor,
                iban_creditor=str(rng.choice(collector_ibans)),
                label=label,
                channel=str(rng.choice(CHANNELS)),
                payment_type=str(rng.choice(PAYMENT_TYPES)),
            )
        )
    return rows


# --------------------------------------------------------------------------
# Point d'entrée principal
# --------------------------------------------------------------------------


def generate(params: GeneratorParams, verbose: bool = False) -> dict[str, pd.DataFrame]:
    """Génère les six tables + `ground_truth` (+ `_debtor_profile` de debug).

    `verbose=True` journalise la progression (utile à grande échelle) —
    silencieux par défaut pour ne pas polluer les tests ni les usages
    normaux du pipeline.

    Retourne un dict de DataFrames prêtes à écrire en Parquet.
    """
    rng = np.random.default_rng(params.seed)

    if verbose:
        _log(f"Génération des cédants ({params.n_assignors:,})...")
    assignors, _ = _gen_parties(rng, params.n_assignors, "ASG", params, is_debtor=False, verbose=verbose)
    if verbose:
        _log(f"Génération des débiteurs ({params.n_debtors:,})...")
    debtors, debtor_profiles = _gen_parties(rng, params.n_debtors, "DBT", params, is_debtor=True, verbose=verbose)
    if verbose:
        _log("Génération des agreements...")
    agreements = _gen_agreements(rng, debtors, debtor_profiles, assignors, params, verbose=verbose)
    if verbose:
        _log(f"Génération des factures ({params.n_invoices:,})...")
    invoices_base = _gen_invoices_base(rng, debtors, agreements, params, verbose=verbose)

    if verbose:
        _log("Construction des plans de règlement...")
    plans = _build_settlement_plans(rng, invoices_base, params.case_weights)

    # dict Python plutôt que DataFrame.loc[] par plan (~des centaines de
    # milliers à plusieurs millions d'appels selon l'échelle) : chaque
    # .loc[] a un coût pandas fixe, dominant à cette fréquence.
    debtors_by_id = debtors.set_index("party_id").to_dict("index")
    profiles_by_id = debtor_profiles.set_index("party_id").to_dict("index")
    assignors_by_id = assignors.set_index("party_id").to_dict("index")
    assignor_id_by_agreement = dict(zip(agreements["agreement_id"], agreements["client_id"]))

    collector_ibans = [_fake_iban(rng) for _ in range(5)]
    technical_ibans = [_fake_iban(rng) for _ in range(5)]
    counters = _Counters()

    all_payments: list[dict] = []
    all_imputations: list[dict] = []
    all_gts: list[dict] = []
    final_amounts: dict[str, int] = {}

    n_plans = len(plans)
    log_every = max(1, n_plans // 20)
    if verbose:
        _log(f"Matérialisation de {n_plans:,} plans de règlement...")
    for i, plan in enumerate(plans):
        if verbose and i > 0 and i % log_every == 0:
            _log(f"  plans : {i:,}/{n_plans:,} ({len(all_payments):,} paiements générés)")
        debtor_row = debtors_by_id[plan["debtor_id"]]
        debtor_profile = profiles_by_id[plan["debtor_id"]]
        agreement_id = plan["invoices"][0]["agreement_id"]
        assignor_id = assignor_id_by_agreement[agreement_id]
        assignor_row = assignors_by_id[assignor_id]

        payments, imputations, gts, finals = _materialize_plan(
            rng,
            plan,
            debtor_row,
            debtor_profile,
            assignor_row["iban"],
            collector_ibans,
            technical_ibans,
            counters,
        )
        all_payments.extend(payments)
        all_imputations.extend(imputations)
        all_gts.extend(gts)
        for f in finals:
            final_amounts[f["invoice_id"]] = f["current_amount"]

    n_orphans = max(1, int(len(invoices_base) * params.case_weights.get("no_invoice", 0.01)))
    all_payments.extend(
        _gen_orphan_payments(
            rng, n_orphans, debtors, assignors, collector_ibans, technical_ibans, params, counters
        )
    )

    invoices = invoices_base.copy()
    invoices["current_amount"] = invoices["invoice_id"].map(final_amounts).fillna(
        invoices["initial_amount"]
    ).astype("int64")
    invoices = invoices.drop(columns=["market"])

    payment_df = pd.DataFrame(all_payments)
    imputation_df = pd.DataFrame(all_imputations)
    ground_truth_df = pd.DataFrame(all_gts)

    payment_df["amount"] = payment_df["amount"].astype("int64")
    invoices["initial_amount"] = invoices["initial_amount"].astype("int64")
    invoices["current_amount"] = invoices["current_amount"].astype("int64")
    if not imputation_df.empty:
        imputation_df["residual_amount"] = imputation_df["residual_amount"].astype("int64")
    if not ground_truth_df.empty:
        ground_truth_df["residual_amount"] = ground_truth_df["residual_amount"].astype("int64")

    return {
        "payment": payment_df.reset_index(drop=True),
        "invoice": invoices.reset_index(drop=True),
        "imputation": imputation_df.reset_index(drop=True),
        "assignor": assignors.reset_index(drop=True),
        "debtor": debtors.reset_index(drop=True),
        "agreement": agreements.reset_index(drop=True),
        "ground_truth": ground_truth_df.reset_index(drop=True),
        "_debtor_profile": debtor_profiles.reset_index(drop=True),
        # `payment` n'a pas de bankroll_code (schéma réel) : le compte
        # technique/de liaison se reconnaît par appartenance à ce petit
        # référentiel d'IBAN connus (config figée côté factor dans la
        # réalité), pas par un champ sur le paiement lui-même.
        "_technical_ibans": pd.DataFrame({"iban": technical_ibans}),
    }
