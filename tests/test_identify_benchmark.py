"""Benchmark of ``identify_objects`` on the held-out ``make_object_signals`` split.

``identify_objects(dataframe)`` screens every uuid for "carries identities of
physical objects or manufacturing entities" and names the object type, from
the values and their behaviour over time only. This file scores it on the
held-out seeds (100-119), whose format variants and hold-time distributions
differ from the dev seeds (0-4) on purpose: tune on the dev split, never on
these.

Scores, pooled over every held-out uuid:

* screening precision and recall of ``role == "identifier"``;
* type accuracy among the correctly screened identifiers, where a predicted
  type counts as correct when it falls in the same group of
  :data:`OBJECT_TYPE_GROUPS` as the label;
* nesting precision and recall: the predicted ``parent_uuid`` edges (child
  -> production parent) against the labelled ones (a serial in its batch, a
  batch in its material). A predicted edge counts when its parent is the
  labelled one.

The result of ``identify_objects`` may be a DataFrame with a ``uuid`` column
(or index) and an ``object_type`` column (``None``/NaN for non-identifiers),
optionally with ``is_identifier`` (bool) or ``role``; or a mapping
``{uuid: object_type or None}``. A predicted type may be a member or a group
name of :data:`OBJECT_TYPE_GROUPS`.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd
import pytest

import ts_shape

#: The type vocabulary, grouped where the data cannot tell the types apart.
#: Every key is a group; a type's own group has the type's name. ``article``,
#: ``rfid_tag`` and ``asset`` are the engine's names (``IDENTIFIER_TYPES``)
#: for a material, a carrier's tag and a machine; the generator never emits
#: them.
OBJECT_TYPE_GROUPS: dict[str, frozenset[str]] = {
    "material": frozenset({"material", "article"}),
    "recipe": frozenset({"recipe"}),
    "batch": frozenset({"batch"}),
    "lot": frozenset({"lot"}),
    "serial": frozenset({"serial"}),
    "work_order": frozenset({"work_order"}),
    "order": frozenset({"sales_order", "purchase_order"}),
    "tooling": frozenset({"tool", "fixture"}),
    "carrier": frozenset({"carrier", "rfid_tag"}),
    "equipment": frozenset({"machine", "station", "asset"}),
    "location": frozenset({"location"}),
    "operator": frozenset({"operator"}),
    "business_partner": frozenset({"customer", "supplier"}),
    "handling_unit": frozenset({"handling_unit"}),
}

HELDOUT_SEEDS = tuple(range(100, 120))
MIN_PRECISION = 0.95
MIN_RECALL = 0.95
MIN_TYPE_ACCURACY = 0.85
MIN_NESTING_PRECISION = 0.8
MIN_NESTING_RECALL = 0.75

_GROUP_OF = {t: g for g, members in OBJECT_TYPE_GROUPS.items() for t in members}
_GROUP_OF.update({g: g for g in OBJECT_TYPE_GROUPS})

#: Groups the engine reports when the data cannot separate their members. A
#: prediction of such a group counts as correct for any of its members.
ENGINE_GROUPS: dict[str, frozenset[str]] = {
    "setup": frozenset({"recipe", "tool", "fixture"}),
}
NOT_IDENTIFIER = "-"


#: Engine vocabulary names that are members of a generator group.
ENGINE_ALIASES = frozenset({"article", "rfid_tag", "asset"})


def test_vocabulary_matches_the_generator():
    from ts_shape._object_signals import OBJECT_TYPES

    members = set().union(*OBJECT_TYPE_GROUPS.values())
    assert members - ENGINE_ALIASES == set(OBJECT_TYPES)
    assert sum(len(m) for m in OBJECT_TYPE_GROUPS.values()) == len(members)


def _identify():
    try:
        return ts_shape.identify_objects
    except (AttributeError, ImportError):
        pytest.skip("ts_shape.identify_objects is not exported yet")


def _missing(value: object) -> bool:
    return value is None or (not isinstance(value, str) and pd.isna(value))


def _predictions(result: object) -> dict[str, str | None]:
    """``{uuid: predicted object type, or None when not an identifier}``."""
    if isinstance(result, Mapping):
        return {
            str(k): None if _missing(v) else str(v).strip().lower()
            for k, v in result.items()
        }
    assert isinstance(result, pd.DataFrame), type(result)
    frame = result if "uuid" in result.columns else result.reset_index()
    assert "uuid" in frame.columns, list(result.columns)
    types = frame["object_type"] if "object_type" in frame.columns else None
    if "is_identifier" in frame.columns:
        flags = frame["is_identifier"].fillna(False).astype(bool)
    elif "role" in frame.columns:
        flags = frame["role"].astype(str).str.lower().eq("identifier")
    else:
        assert types is not None, list(frame.columns)
        flags = types.map(lambda v: not _missing(v))
    out: dict[str, str | None] = {}
    for i, uuid in enumerate(frame["uuid"].astype(str)):
        if not flags.iloc[i]:
            out[uuid] = None
            continue
        t = None if types is None else types.iloc[i]
        out[uuid] = "unknown" if _missing(t) else str(t).strip().lower()
    return out


def benchmark_data(seeds=HELDOUT_SEEDS, days: float = 3):
    """``[(seed, frame, labels)]`` of the benchmark seeds."""
    return [
        (seed, *ts_shape.make_object_signals(seed=seed, days=days)) for seed in seeds
    ]


def score(identify, data) -> dict[str, object]:
    """Screening and typing scores of ``identify`` on ``benchmark_data()``."""
    rows = []
    for seed, frame, labels in data:
        predicted = _predictions(identify(frame))
        for label in labels.itertuples():
            pred = predicted.get(label.uuid)
            truth = label.object_type if label.role == "identifier" else None
            rows.append(
                {
                    "seed": seed,
                    "uuid": label.uuid,
                    "variant": label.variant,
                    "truth": truth or f"<{label.role}>",
                    "true_id": truth is not None,
                    "pred_id": pred is not None,
                    "true_group": _GROUP_OF.get(truth) if truth else None,
                    "pred_group": _GROUP_OF.get(pred, pred) if pred else None,
                }
            )
    table = pd.DataFrame(rows)
    tp = int((table["true_id"] & table["pred_id"]).sum())
    screened = table.loc[table["true_id"] & table["pred_id"]]
    in_engine_group = pd.Series(
        [
            truth in ENGINE_GROUPS.get(pred, ())
            for truth, pred in zip(screened["truth"], screened["pred_group"])
        ],
        index=screened.index,
        dtype=bool,
    )
    hits = (screened["true_group"] == screened["pred_group"]) | in_engine_group
    return {
        "precision": tp / max(int(table["pred_id"].sum()), 1),
        "recall": tp / max(int(table["true_id"].sum()), 1),
        "type_accuracy": float(hits.mean()) if len(screened) else 0.0,
        "confusion": pd.crosstab(
            table["truth"], table["pred_group"].fillna(NOT_IDENTIFIER)
        ),
        "table": table,
    }


def _parents(result: object) -> dict[str, str]:
    """``{child uuid: predicted parent uuid}`` (no entry: no parent)."""
    if not isinstance(result, pd.DataFrame) or "parent_uuid" not in result.columns:
        return {}
    frame = result if "uuid" in result.columns else result.reset_index()
    return {
        str(u): str(p)
        for u, p in zip(frame["uuid"], frame["parent_uuid"], strict=True)
        if not _missing(p)
    }


def nesting_score(identify, data) -> dict[str, object]:
    """Nesting precision and recall of ``identify`` on ``benchmark_data()``."""
    rows = []
    for seed, frame, labels in data:
        result = identify(frame)
        table = getattr(result, "table", result)
        predicted = _parents(table)
        truth = {
            str(u): str(p)
            for u, p in zip(labels["uuid"], labels["parent_uuid"], strict=True)
            if not _missing(p)
        }
        for child in sorted(set(predicted) | set(truth)):
            rows.append(
                {
                    "seed": seed,
                    "child": child,
                    "predicted": predicted.get(child),
                    "truth": truth.get(child),
                }
            )
    table = pd.DataFrame(rows, columns=["seed", "child", "predicted", "truth"])
    hit = table["predicted"].notna() & (table["predicted"] == table["truth"])
    return {
        "precision": int(hit.sum()) / max(int(table["predicted"].notna().sum()), 1),
        "recall": int(hit.sum()) / max(int(table["truth"].notna().sum()), 1),
        "table": table,
    }


@pytest.fixture(scope="module")
def heldout_results():
    """The held-out data and one ``identify_objects`` result per frame."""
    identify = _identify()
    data = benchmark_data()
    results = {id(frame): identify(frame) for _, frame, _ in data}
    return data, lambda frame: results[id(frame)]


@pytest.fixture(scope="module")
def heldout_scores(heldout_results):
    return score(heldout_results[1], heldout_results[0])


@pytest.fixture(scope="module")
def heldout_nesting(heldout_results):
    return nesting_score(heldout_results[1], heldout_results[0])


def _report(scores: dict[str, object]) -> str:
    table = scores["table"]
    assert isinstance(table, pd.DataFrame)
    wrong = table.loc[
        (table["true_id"] != table["pred_id"])
        | (
            table["true_id"]
            & (table["true_group"] != table["pred_group"])
            & ~pd.Series(
                [
                    t in ENGINE_GROUPS.get(p, ())
                    for t, p in zip(table["truth"], table["pred_group"])
                ],
                index=table.index,
                dtype=bool,
            )
        )
    ]
    with pd.option_context("display.width", 250, "display.max_columns", 50):
        return (
            f"precision={scores['precision']:.3f} recall={scores['recall']:.3f} "
            f"type_accuracy={scores['type_accuracy']:.3f}\n"
            f"confusion (rows: label, columns: predicted group):\n"
            f"{scores['confusion'].to_string()}\n"
            f"misses by variant:\n"
            f"{wrong.groupby(['truth', 'variant']).size().to_string()}"
        )


def test_screening_precision(heldout_scores):
    assert heldout_scores["precision"] >= MIN_PRECISION, _report(heldout_scores)


def test_screening_recall(heldout_scores):
    assert heldout_scores["recall"] >= MIN_RECALL, _report(heldout_scores)


def test_type_accuracy(heldout_scores):
    assert heldout_scores["type_accuracy"] >= MIN_TYPE_ACCURACY, _report(heldout_scores)


def _nesting_report(scores: dict[str, object]) -> str:
    table = scores["table"]
    assert isinstance(table, pd.DataFrame)
    wrong = table[table["predicted"] != table["truth"]]
    with pd.option_context("display.width", 250, "display.max_rows", 200):
        return (
            f"nesting precision={scores['precision']:.3f} "
            f"recall={scores['recall']:.3f}\nwrong edges:\n{wrong.to_string()}"
        )


def test_nesting_precision(heldout_nesting):
    assert heldout_nesting["precision"] >= MIN_NESTING_PRECISION, _nesting_report(
        heldout_nesting
    )


def test_nesting_recall(heldout_nesting):
    assert heldout_nesting["recall"] >= MIN_NESTING_RECALL, _nesting_report(
        heldout_nesting
    )


def test_scoring_rewards_the_labels_and_punishes_guessing():
    """The scorer itself: an oracle scores 1.0, calling everything a material
    finds every identifier but has poor precision and typing."""
    data = benchmark_data(seeds=(100, 101), days=1)
    labels_of = {id(frame): labels for _, frame, labels in data}

    def oracle(frame):
        return labels_of[id(frame)][["uuid", "object_type"]]

    def everything(frame):
        return {u: "material" for u in frame["uuid"].unique()}

    perfect = score(oracle, data)
    assert (perfect["precision"], perfect["recall"]) == (1.0, 1.0)
    assert perfect["type_accuracy"] == 1.0
    nested = nesting_score(lambda frame: labels_of[id(frame)], data)
    assert (nested["precision"], nested["recall"]) == (1.0, 1.0)
    assert len(nested["table"]) > 0

    # an edge to the wrong parent is wrong; no edge finds nothing
    def shifted(frame):
        labels = labels_of[id(frame)].copy()
        labels["parent_uuid"] = labels["parent_uuid"].map(
            lambda p: None if p is None else p + "x"
        )
        return labels

    off = nesting_score(shifted, data)
    assert (off["precision"], off["recall"]) == (0.0, 0.0)
    none = nesting_score(lambda frame: oracle(frame), data)
    assert none["recall"] == 0.0
    naive = score(everything, data)
    assert naive["recall"] == 1.0
    assert naive["precision"] < 0.7 and naive["type_accuracy"] < 0.2
    assert NOT_IDENTIFIER not in naive["confusion"].columns
    # a group name or any member of the group counts as the same type
    assert _GROUP_OF["supplier"] == _GROUP_OF["customer"] == "business_partner"
    assert _GROUP_OF["order"] == _GROUP_OF["purchase_order"] == "order"
