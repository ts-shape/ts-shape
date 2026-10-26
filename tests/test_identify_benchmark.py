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
  labelled one;
* the same nesting scores on the held-out seeds generated with every
  material polled every 30 minutes (``material_poll``, opt-in), where a
  material change is seen up to half an hour after the batch change it
  goes with;
* the nesting scores of the edges into a material on the held-out seeds
  generated with materials run in campaigns that change about once a day
  (``material_changes_per_day``, opt-in): one to three changes in three
  days;
* on the held-out seeds generated with the discrete options
  (``DISCRETE_OPTIONS``: a door per line bracketing each serial scan, held
  inputs written with ``is_delta`` and a keep-alive, same-value re-setups,
  one line): the cycle signal of each serial (``cycle_uuid``), the usage
  links with their qualifier (``uses``; also with the types pinned to the
  labels, which isolates the link logic from typing errors), and the
  screening and typing of the held inputs.

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
MIN_TYPE_ACCURACY = 0.87
MIN_NESTING_PRECISION = 0.92
MIN_NESTING_RECALL = 0.92
#: Material polling period (seconds) of the coarse-material benchmark, and
#: its nesting thresholds.
COARSE_MATERIAL_POLL = 1800.0
MIN_COARSE_NESTING_PRECISION = 0.8
MIN_COARSE_NESTING_RECALL = 0.8
#: Material changes per day of the rare-material benchmark, and the nesting
#: thresholds of its edges into a material.
RARE_MATERIAL_CHANGES_PER_DAY = 1.0
MIN_RARE_NESTING_PRECISION = 0.85
MIN_RARE_NESTING_RECALL = 0.85

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


def benchmark_data(seeds=HELDOUT_SEEDS, days: float = 3, **options):
    """``[(seed, frame, labels)]`` of the benchmark seeds; ``options`` are
    the generator's opt-in options (``material_poll`` polls every material
    on a grid of that period in seconds, ``material_changes_per_day`` runs
    the materials in campaigns)."""
    if not options:
        return [
            (seed, *ts_shape.make_object_signals(seed=seed, days=days))
            for seed in seeds
        ]
    from ts_shape._object_signals import generate

    return [
        (seed, *generate(seed=seed, days=days, include=None, noise=True, **options))
        for seed in seeds
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


def nesting_score(identify, data, parent_type: str | None = None) -> dict[str, object]:
    """Nesting precision and recall of ``identify`` on ``benchmark_data()``;
    with ``parent_type``, of the edges into a parent labelled that type
    only (labelled, or predicted to a signal of that type)."""
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
        children = set(predicted) | set(truth)
        if parent_type is not None:
            of_type = set(labels.loc[labels["object_type"] == parent_type, "uuid"])
            children = {
                c
                for c in children
                if truth.get(c) in of_type or predicted.get(c) in of_type
            }
        for child in sorted(children):
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


@pytest.fixture(scope="module")
def coarse_nesting():
    """Nesting scores on the held-out seeds with coarsely polled materials."""
    identify = _identify()
    data = benchmark_data(material_poll=COARSE_MATERIAL_POLL)
    return nesting_score(identify, data)


def test_coarse_material_nesting(coarse_nesting):
    # a material polled every 30 min is seen to change up to half an hour
    # after its batch does: the batch still nests in it
    assert coarse_nesting["precision"] >= MIN_COARSE_NESTING_PRECISION, _nesting_report(
        coarse_nesting
    )
    assert coarse_nesting["recall"] >= MIN_COARSE_NESTING_RECALL, _nesting_report(
        coarse_nesting
    )


@pytest.fixture(scope="module")
def rare_nesting():
    """Nesting scores of the edges into a material on the held-out seeds with
    materials that change about once a day."""
    identify = _identify()
    data = benchmark_data(material_changes_per_day=RARE_MATERIAL_CHANGES_PER_DAY)
    return nesting_score(identify, data, parent_type="material")


def test_rare_material_nesting(rare_nesting):
    # a material changed about once a day changes one to three times in
    # three days: every change meets a batch change, all of them together
    # far less likely by chance than one in a hundred over the pairs tried
    assert rare_nesting["precision"] >= MIN_RARE_NESTING_PRECISION, _nesting_report(
        rare_nesting
    )
    assert rare_nesting["recall"] >= MIN_RARE_NESTING_RECALL, _nesting_report(
        rare_nesting
    )


def _digest(frame: pd.DataFrame) -> str:
    """A digest of the values of ``frame`` (timestamps as nanoseconds)."""
    import hashlib

    h = hashlib.sha256()
    for column in frame.columns:
        values = frame[column]
        if column == "systime":
            values = values.astype("datetime64[ns]").astype("int64")
        cells = values.astype(object).where(values.notna(), None).tolist()
        h.update(str(column).encode())
        h.update("\x1f".join(map(repr, cells)).encode())
    return h.hexdigest()[:16]


@pytest.mark.parametrize(
    ("seed", "material_poll", "frame_digest", "labels_digest"),
    [
        # the data of the generator before the opt-in options were added
        (0, None, "7ee2ba682dce29dc", "27bfa15349b329a5"),
        (101, None, "f3f81b0edcb68a56", "4eec7681a4a7eabd"),
        (101, 1800.0, "ef699cbf3c2391df", "4eec7681a4a7eabd"),
    ],
)
def test_generator_options_keep_the_default_data(
    seed, material_poll, frame_digest, labels_digest
):
    from ts_shape._object_signals import generate

    frame, labels = generate(
        seed=seed, days=1, include=None, noise=True, material_poll=material_poll
    )
    assert (_digest(frame), _digest(labels)) == (frame_digest, labels_digest)
    same = generate(
        seed=seed,
        days=1,
        include=None,
        noise=True,
        material_poll=material_poll,
        material_changes_per_day=None,
        continuous_process=None,
        discrete_cycles=False,
        constant_materials=0,
        resetup_same_value=False,
        lines=None,
    )
    pd.testing.assert_frame_equal(frame, same[0])
    pd.testing.assert_frame_equal(labels, same[1])


def test_material_campaigns_change_rarely():
    from ts_shape._object_signals import generate

    frame, labels = generate(
        seed=101, days=3, include=None, noise=False, material_changes_per_day=1.0
    )
    materials = labels.loc[labels["object_type"] == "material", "uuid"]
    assert len(materials)
    for uuid in materials:
        rows = frame[frame["uuid"] == uuid]
        column = labels.set_index("uuid").loc[uuid, "value_column"]
        values = rows[column].dropna()
        assert 1 <= int((values != values.shift()).sum()) - 1 <= 6
    # every batch nests in its material
    batches = labels[labels["object_type"] == "batch"]
    assert batches["parent_uuid"].isin(materials).all()
    for kind in ("lot", "serial"):
        cont, cont_labels = generate(
            seed=101, days=3, include=None, noise=False, continuous_process=kind
        )
        types = set(cont_labels["object_type"].dropna())
        assert "batch" not in types and "work_order" not in types
        mats = set(cont_labels.loc[cont_labels["object_type"] == "material", "uuid"])
        nested = cont_labels[cont_labels["object_type"] == kind]
        assert nested["parent_uuid"].isin(mats).all()
    with pytest.raises(ValueError, match="continuous_process"):
        generate(seed=0, days=1, include=None, noise=True, continuous_process="x")
    with pytest.raises(ValueError, match="material_changes_per_day"):
        generate(seed=0, days=1, include=None, noise=True, material_changes_per_day=0.0)


def _edges(table: pd.DataFrame, column: str) -> set[tuple[str, ...]]:
    """``(child, parent)`` edges of ``column`` (``cycle_uuid``), or ``(child,
    parent, qualifier)`` of ``uses``."""
    out: set[tuple[str, ...]] = set()
    if column not in table.columns:
        return out
    for uuid, value in zip(table["uuid"], table[column], strict=True):
        if column == "uses":
            out |= {(str(uuid), str(p), str(q)) for p, q in value or ()}
        elif not _missing(value):
            out.add((str(uuid), str(value)))
    return out


def edge_score(results, column: str) -> dict[str, float]:
    """Precision and recall of the ``column`` edges over ``[(labels,
    table)]``."""
    tp = predicted = true = 0
    for labels, table in results:
        truth, got = _edges(labels, column), _edges(table, column)
        tp += len(truth & got)
        predicted += len(got)
        true += len(truth)
    return {"precision": tp / max(predicted, 1), "recall": tp / max(true, 1)}


#: The discrete benchmark (opt-in generator options): a door per line
#: bracketing each serial scan, two held inputs per line written with
#: is_delta and a 2 s keep-alive (plus a setpoint, a state text and a
#: configuration text written the same way), same-value re-setups, one line.
DISCRETE_OPTIONS: dict[str, object] = {
    "discrete_cycles": True,
    "constant_materials": 2,
    "resetup_same_value": True,
    "lines": 1,
}
MIN_CYCLE_PRECISION = 0.95
MIN_CYCLE_RECALL = 0.95
#: Usage links, qualifier included: measured 0.86 / 0.85 held-out (the
#: misses come from screening, typing and nesting); with the types pinned to
#: the labels 0.97 / 0.99.
MIN_USES_PRECISION = 0.82
MIN_USES_RECALL = 0.82
MIN_PINNED_USES_PRECISION = 0.95
MIN_PINNED_USES_RECALL = 0.95
MIN_HELD_INPUTS = 0.95


def _pinned(labels: pd.DataFrame) -> dict[str, str]:
    """The labelled types as hints (the generator's ``machine`` is the
    engine's ``asset``)."""
    return {
        str(u): {"machine": "asset"}.get(str(t), str(t))
        for u, r, t in zip(labels["uuid"], labels["role"], labels["object_type"])
        if r == "identifier" and not _missing(t)
    }


@pytest.fixture(scope="module")
def discrete_results():
    """``(plain, pinned)``: ``[(labels, table)]`` of ``identify_objects`` on
    the held-out discrete benchmark, without and with the types pinned."""
    identify = _identify()
    data = benchmark_data(**DISCRETE_OPTIONS)
    plain = [(labels, identify(frame).table) for _, frame, labels in data]
    pinned = [
        (labels, identify(frame, hints=_pinned(labels)).table)
        for _, frame, labels in data
    ]
    return plain, pinned


def test_cycle_detection(discrete_results):
    got = edge_score(discrete_results[0], "cycle_uuid")
    assert got["precision"] >= MIN_CYCLE_PRECISION, got
    assert got["recall"] >= MIN_CYCLE_RECALL, got


def test_usage_links(discrete_results):
    got = edge_score(discrete_results[0], "uses")
    assert got["precision"] >= MIN_USES_PRECISION, got
    assert got["recall"] >= MIN_USES_RECALL, got
    pinned = edge_score(discrete_results[1], "uses")
    assert pinned["precision"] >= MIN_PINNED_USES_PRECISION, pinned
    assert pinned["recall"] >= MIN_PINNED_USES_RECALL, pinned


def test_held_inputs_are_typed_materials(discrete_results):
    rows = []
    for labels, table in discrete_results[0]:
        merged = labels.merge(table, on="uuid", suffixes=("", "_got"))
        rows.append(merged)
    both = pd.concat(rows, ignore_index=True)
    held = both[both["variant"].str.startswith("held_input")]
    assert len(held)
    assert (held["role_got"] == "identifier").mean() >= MIN_HELD_INPUTS
    typed = held["object_type_got"].map(lambda t: _GROUP_OF.get(t, t))
    assert (typed == "material").mean() >= MIN_HELD_INPUTS
    settings = both[both["variant"].str.startswith("held_setting")]
    assert len(settings) and (settings["role_got"] != "identifier").all()


def test_discrete_options_label_cycles_and_uses():
    from ts_shape._object_signals import generate

    frame, labels = generate(
        seed=101, days=1, include=None, noise=False, **DISCRETE_OPTIONS
    )
    assert list(labels.columns)[-2:] == ["cycle_uuid", "uses"]
    serials = labels[labels["object_type"] == "serial"]
    assert serials["cycle_uuid"].notna().all()
    doors = set(serials["cycle_uuid"])
    assert set(frame.loc[frame["uuid"].isin(doors), "value_bool"].dropna()) == {
        True,
        False,
    }
    qualifiers = {q for uses in serials["uses"] for _, q in uses}
    assert qualifiers <= {"made_from", "made_with", "made_by", "carried_by"}
    assert "made_from" in qualifiers
    held = labels.loc[labels["variant"].str.startswith("held_input"), "uuid"]
    assert len(held) == 2
    rows = frame[frame["uuid"] == held.iloc[0]]
    assert rows["is_delta"].any() and not rows["is_delta"].all()
    with pytest.raises(ValueError, match="lines"):
        generate(seed=0, days=1, include=None, noise=True, lines=3)
    with pytest.raises(ValueError, match="constant_materials"):
        generate(seed=0, days=1, include=None, noise=True, constant_materials=-1)


def test_material_poll_keeps_the_default_data():
    from ts_shape._object_signals import generate

    frame, labels = ts_shape.make_object_signals(seed=101, days=1)
    same, same_labels = generate(
        seed=101, days=1, include=None, noise=True, material_poll=None
    )
    pd.testing.assert_frame_equal(frame, same)
    pd.testing.assert_frame_equal(labels, same_labels)
    polled, polled_labels = generate(
        seed=101, days=1, include=None, noise=True, material_poll=1800.0
    )
    materials = polled_labels.loc[polled_labels["object_type"] == "material", "uuid"]
    assert len(materials)
    for uuid in materials:
        gaps = polled.loc[polled["uuid"] == uuid, "systime"].diff().dropna()
        assert gaps.dt.total_seconds().median() == pytest.approx(1800, rel=0.05)


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
