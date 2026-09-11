# Moteur de rapprochement automatique paiement / facture

Moteur de rapprochement (lettrage) automatique pour une activité d'affacturage. Génère des données synthétiques, entraîne un modèle de scoring de paires paiement/facture, reconstitue les groupes de règlement (1↔1, 1↔n, n↔1, n↔n), applique des seuils de décision calibrés, et peut tourner en batch quotidien.

La spécification fonctionnelle complète est dans [`spec_rapprochement_automatique.md`](spec_rapprochement_automatique.md) — ce README ne la remplace pas, il explique comment faire tourner et modifier le code qui l'implémente.

## Architecture

Quatre étages (§1 de la spec), dans l'ordre où le paiement les traverse :

| Étage | Module | Rôle |
|---|---|---|
| A — Génération de candidats | `src/blocking.py` | Réduit l'espace de recherche par règles dures (référence, montant, IBAN, nom) |
| B — Scoring de paires | `src/model.py`, `src/features.py` | LightGBM à deux passes, calibration isotonique |
| C — Résolution d'ensembles | `src/subsets.py` | Reconstitue les groupes 1↔n / n↔1 / n↔n |
| D — Décision | `src/decision.py` | Seuils par segment + condition de marge → auto-validation / revue / rejet |

Le tout repose sur un principe unique : **le rapprochement est un problème événementiel**. `src/events.py` dérive un journal ordonné des six tables sources, et `src/state.py` (`LedgerState`) reconstruit l'état du grand livre à un instant `t` donné, en n'utilisant jamais d'information postérieure à `t`. Toute feature, tout candidat, passe par cette interface — c'est la garantie mécanique d'absence de fuite (§8.2 de la spec).

### Écarts connus avec le schéma réel

Le générateur synthétique (`src/generator.py`) s'écarte du schéma §2 de la spec sur deux points, alignés sur le schéma réel observé :

- **`debtor` n'a pas de `closed_at`** (seul `opened_at` existe) — un débiteur, une fois ouvert, ne se ferme jamais. Seul `assignor` porte les deux dates.
- **`payment` n'a pas de `bankroll_code`** — résolu par jointure sur le débiteur de la facture candidate (`lookups["debtor_bankroll_code"]`, voir `src/blocking.py::build_static_lookups`). Le routage IBAN vers un compte technique/de liaison (`TECHNICAL_ACCOUNT`) ne dépend donc plus d'un champ sur le paiement mais de l'appartenance de son IBAN à un petit référentiel `_technical_ibans.parquet` (généré par `generator.py`, config figée côté factor dans la réalité).

## Structure du projet

```
src/
    generator.py   Génération de données synthétiques (Phase 0)
    events.py      Journal d'événements ordonné (§8.1)
    state.py       LedgerState — état incrémental, jamais de fuite (§8.2)
    normalize.py   Normalisation libellé / référence (§3.1)
    blocking.py    Étage A — candidats + résolution IBAN (§3.2, §4)
    baseline.py    Rapprochement par règles, sans apprentissage — le plancher de comparaison
    features.py    Featurisation des paires (§5.3, toutes familles)
    replay.py      Rejeu de l'historique → dataset d'entraînement (§8.3)
    model.py       Étage B — LightGBM, split temporel, 2 passes, calibration (§5, §7.1)
    subsets.py     Étage C — résolution d'ensembles (§6)
    decision.py    Étage D — seuils, marge, segments (§7.1)
    batch.py       Orchestration production + backtest (§8.4, §8.5)
scripts/           Points d'entrée CLI (un par phase, voir plus bas)
tests/             Tests pytest, un fichier par module de src/
data/              Parquet généré — non versionné, régénérable
models/            Modèles entraînés (LightGBM + calibrateur) — régénérable
```

## Installation

Python 3.11+.

```bash
pip install -r requirements.txt
```

Dépendances principales : `pandas`, `numpy`, `pyarrow` (Parquet), `rapidfuzz` (similarité de texte), `lightgbm`, `scikit-learn`, `scipy`, `joblib`, `pytest`.

> Sous WSL/Linux, LightGBM a besoin de la lib OpenMP système : `sudo apt-get install -y libgomp1` si vous obtenez `OSError: libgomp.so.1`.

## Utiliser le projet — pipeline complet

Chaque script suppose que les étapes précédentes ont déjà tourné (chacun affiche en tête de fichier les prérequis). Depuis la racine du dépôt :

```bash
# 1. Génère les données synthétiques (data/*.parquet)
python scripts/generate_data.py

# 2. Mesure le rappel de l'étage A (candidats) — doit rester >= 99%
python scripts/measure_blocking_recall.py

# 3. Baseline par règles — le chiffre à battre par le modèle
python scripts/evaluate_baseline.py

# 4. Rejoue l'historique, construit le dataset d'entraînement (data/dataset/)
python scripts/build_dataset.py

# 5. Entraîne le modèle (2 passes), compare à la baseline (models/*.joblib)
python scripts/train_model.py

# 6. Évalue la reconstruction de groupes (étage C) sur le test set
python scripts/evaluate_subsets.py

# 7. Évalue les seuils de décision (étage D) : courbe automatisation/précision
python scripts/evaluate_decision.py

# 8. Backtest complet (étages A-D) sur les 2 derniers mois, jamais vus
python scripts/run_backtest.py
```

Script bonus, indépendant du reste : `python scripts/replay_demo.py` rejoue le journal et affiche l'état reconstruit (utile pour comprendre `LedgerState` sans lancer tout le pipeline).

### Tests

```bash
python -m pytest tests/ -q
```

~133 tests, tous doivent passer avant de considérer un changement terminé. Ils tournent sur des jeux de données **générés à la volée** (petits volumes, seeds fixes) — aucun test ne dépend du contenu de `data/`.

### Régénérer après une modification

Si vous touchez à `generator.py` → tout le reste doit être régénéré dans l'ordre (étapes 1 à 8 ci-dessus). Si vous touchez seulement à `features.py`, `model.py`, `subsets.py` ou `decision.py` → repartez de l'étape 4 (le dataset de features doit être reconstruit) ou de l'étape 5 si seul le modèle change.

## Performance à grande échelle

Testé et validé jusqu'à 200 000 débiteurs / ~31 700 paiements en local (au-delà, la génération synthétique elle-même — pas le pipeline — devient le facteur limitant). Le point sensible était l'étage A : comparer chaque paiement à *tous* les débiteurs (clé K4, similarité de nom) est O(n_paiements × n_débiteurs) — infaisable au-delà de quelques dizaines de milliers de débiteurs (des jours de calcul à partir du million). `src/blocking.py::build_static_lookups` construit à la place un **index inversé** par token de nom, avec un **élagage dynamique des tokens trop fréquents** (`_NAME_INDEX_MIN_BUCKET` / `_NAME_INDEX_MAX_BUCKET_FRACTION`) — une liste de stopwords figée ne suffit pas, un mot de secteur ("INDUSTRIE", "SERVICES"...) peut être partagé par une grosse fraction des débiteurs réels sans qu'on puisse l'anticiper. Résultat mesuré (`blocking + featurisation`, `build_training_rows`) : temps par paiement quasiment stable entre 200 et 200 000 débiteurs (~1 à 4,5 ms/paiement) — extrapolé à 500 000 paiements, de l'ordre de 30 à 40 minutes, contre plusieurs centaines de jours avant correctif.

Si une nouvelle lenteur apparaît à l'échelle réelle (2M débiteurs, 500K paiements, ~8M événements) :
1. Vérifiez d'abord que ce n'est pas à nouveau une clé de blocking non indexée (K1/K2/K3 utilisent déjà les index de `LedgerState`, K4 l'index inversé ci-dessus — toute nouvelle clé doit suivre le même principe, jamais une boucle sur l'ensemble des débiteurs/factures).
2. `LedgerState` (`state.py`) reste la partie la plus sensible : `dataclasses.asdict()` est appelé à chaque lecture de facture (`get_invoice`/`open_invoices`), ce n'est pas encore optimisé pour un très gros volume — un remplacement par une construction de dict manuelle serait le prochain gain le plus probable si le profilage le confirme.
3. `features.py::featurize` reste une fonction par paire (candidat), appelée en boucle Python dans `replay.py`/`batch.py` — vectoriser la featurisation (pandas/numpy sur tout le lot d'un paiement, `rapidfuzz.process.cdist` pour la similarité de nom en lot) est le levier suivant si le volume de candidats par paiement grossit fortement.

## Comment modifier le projet

### Ajouter ou changer une feature (§5.3)

Toutes les features vivent dans `src/features.py`, regroupées par famille (`_amount_features`, `_temporal_features`, `_textual_features`, `_identity_features`, `_behavioral_features`, `_context_features`). `featurize()`, `generate_candidates()`, `resolve_iban()` et `baseline_match()` prennent tous un unique paramètre `lookups: dict` (produit par `blocking.build_static_lookups`) plutôt que des arguments individuels — si une feature a besoin d'un nouveau référentiel statique (IBAN, table de correspondance...), ajoutez-le comme clé de ce dict plutôt que de faire grossir les signatures.

Pour ajouter une feature :

1. Écrivez-la dans la fonction de famille appropriée (ou créez-en une nouvelle).
2. Si elle dépend du temps ou de l'historique, elle **doit** passer par une méthode de `LedgerState` avec un `as_of` explicite — jamais lire `tables['invoice']`/`tables['imputation']` directement dans `features.py`. C'est la règle non négociable du projet (voir `tests/test_replay.py::test_removing_future_events_does_not_change_features` et `test_amount_features_use_state_not_final_invoice_balance` pour le genre de test qui doit continuer à passer).
3. Si elle est catégorielle, ajoutez son nom à `CATEGORICAL_COLUMNS` dans `src/model.py`.
4. Régénérez le dataset (`build_dataset.py`) et réentraînez (`train_model.py`).
5. Ajoutez un test dans `tests/test_features.py` qui construit un petit état à la main (voir les fixtures existantes) et vérifie la valeur calculée.

### Changer le comportement du générateur

`src/generator.py` contrôle les proportions de cas (`DEFAULT_CASE_WEIGHTS`), les profils de débiteurs, le bruit sur les libellés. Après modification, régénérez (`generate_data.py`) et relancez toute la chaîne — les nombres rapportés par les autres scripts n'ont de sens que sur des données fraîches.

### Changer le modèle (étage B)

`src/model.py` contient le split temporel (`temporal_split`, bornes `TRAIN_START`/`TRAIN_END`/`VAL_END`/`TEST_END`), l'entraînement (`train_model`), l'architecture deux passes (`train_two_pass`, `out_of_fold_scores`, `add_competition_features`). Le split est calé sur les 14 mois du générateur par défaut — si vous changez `n_months` dans le générateur, ajustez ces constantes en conséquence (et gardez 2 mois de côté, non utilisés en train/val/test, pour le backtest de `run_backtest.py`).

### Changer la résolution d'ensembles (étage C)

`src/subsets.py` : `find_subsets` fait la recherche combinatoire bornée (DFS avec budget de nœuds). `propose_groups` orchestre 3 passes (somme exacte → agrégat n↔n → candidat unique en dernier recours) — **l'ordre compte**, voir les commentaires dans le code, un mauvais ordre fait disparaître silencieusement les agrégats n↔n. `resolve_conflicts` ne bloque une facture que si une proposition la clôture *totalement* (`is_full=True`) — un versement partiel ne doit jamais être bloqué par une clôture antérieure ou postérieure sur la même facture.

### Changer les seuils de décision (étage D)

`src/decision.py` : `fit_segment_thresholds` calibre un seuil par segment si le volume de validation le permet (`min_segment_samples`), sinon retombe sur le seuil global. Si vous ajoutez une dimension de segmentation, passez-la dans `segment_columns` (liste de noms de colonnes déjà présentes dans le dataframe top-1) — aucun autre changement de code nécessaire.

### Ajouter une étape au batch de production

`src/batch.py::run_backtest` est la boucle jour par jour (§8.4). Le point délicat : l'état n'avance **qu'une fois par jour** (pas paiement par paiement comme dans `replay.py`), pour que deux paiements du même jour soient arbitrés ensemble par `resolve_conflicts`. Si vous ajoutez une étape (ex. réinjection des décisions humaines, §7.2), insérez-la à la fin de la boucle `while current_day <= end_day`, avant l'expiration du reliquat.

## Points de vigilance (déjà réglés, à ne pas casser)

- **`current_amount`** : pas de champ `imputed_amount` dans le schéma `imputation` (§2.3). La convention retenue (documentée en tête de `src/state.py`) : `residual_amount` est le solde *après* la ligne — `FULL` solde à 0, `PARTIAL` laisse `residual_amount`. Toute nouvelle logique touchant aux montants doit respecter cette convention.
- **Ordonnancement du journal** : `value_date` sert à ordonner le journal (pas de champ de comptabilisation distinct dans le schéma). Départage des ex æquo par priorité de type d'événement puis identifiant (`src/events.py::EVENT_TYPE_PRIORITY`) — ne pas réordonner sans comprendre l'impact sur `test_state.py`.
- **`as_of` obligatoire** : toute méthode de lecture de `LedgerState` l'exige et refuse une valeur antérieure à son horloge interne. C'est volontaire — ne contournez pas avec `state._invoices` directement en dehors des tests.
- **Négatifs non échantillonnés** : le dataset d'entraînement garde tous les candidats de l'étage A comme négatifs (sauf le positif) — ne pas sous-échantillonner, c'est ce qui donne au modèle des négatifs difficiles (§5.1).
