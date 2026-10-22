"""Labelled benchmark of object-identifier signals (``make_object_signals``).

The public entry point is :func:`ts_shape.datasets.make_object_signals`; this
private module holds the simulation. One or two production lines plus a
plant level (shipping, goods receipt, warehouse) are simulated: work orders
with setups, stops and cycles, batches inside the material, one serial per
cycle, carriers circulating in a loop, shift-bound operators, and so on.
Every signal is then sampled the way a historian stores it (on change, with
heartbeats, or periodically, with gaps) and optionally made noisy.

Format variants are tagged ``"all"`` or ``"heldout"``. Seeds below
:data:`HELDOUT_SEED_MIN` (the dev split) only draw ``"all"`` variants and
narrower hold distributions; held-out seeds draw from every variant, prefer
the held-out ones and use wider, differently shaped hold distributions.
"""

from __future__ import annotations

import uuid as _uuid
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import pandas as pd

from ts_shape.loader._utils import empty_signal_frame

STR, INT, DBL = "value_string", "value_integer", "value_double"

OBJECT_TYPES: tuple[str, ...] = (
    "material",
    "recipe",
    "batch",
    "lot",
    "serial",
    "work_order",
    "sales_order",
    "purchase_order",
    "tool",
    "fixture",
    "carrier",
    "machine",
    "station",
    "location",
    "operator",
    "customer",
    "supplier",
    "handling_unit",
)
ROLES: tuple[str, ...] = (
    "identifier",
    "state",
    "free_text",
    "measurement",
    "counter",
    "timestamp",
    "constant",
)
DEV_SEEDS: tuple[int, ...] = tuple(range(5))
HELDOUT_SEEDS: tuple[int, ...] = tuple(range(100, 120))
HELDOUT_SEED_MIN = 100

#: ``{object_type: {variant: (split, value_column)}}``; split is ``"all"``
#: (dev and held-out seeds) or ``"heldout"`` (held-out seeds only).
FORMAT_VARIANTS: dict[str, dict[str, tuple[str, str]]] = {
    "material": {
        "sap18": ("all", STR),  # 000000000010423317
        "mat_prefix": ("all", STR),  # MAT-104233
        "gtin13": ("all", STR),  # 4012345123452
        "gtin14": ("heldout", STR),  # 14012345123459
        "sap_int": ("heldout", INT),  # 10423317
        "alnum_part": ("heldout", STR),  # 3KA5130-1AB
    },
    "recipe": {
        "rz_name": ("all", STR),  # RZ_PA66_GF30_v3
        "prg": ("all", STR),  # PRG017
        "program_int": ("heldout", INT),  # 17
        "variant_code": ("heldout", STR),  # HSG-120-BK
    },
    "batch": {
        "date_seq": ("all", STR),  # 240315-07
        "week_seq": ("all", STR),  # B24W11-003
        "gs1_10": ("all", STR),  # (10)240315A07
        "julian": ("heldout", STR),  # 24075-1-012
        "int_date_seq": ("heldout", INT),  # 24031507
    },
    "lot": {
        "supplier_alnum": ("all", STR),  # L2403A17
        "charge": ("all", STR),  # CH-58812-2
        "gs1_10_lot": ("heldout", STR),  # (10)KX48812
        "numeric_lot": ("heldout", STR),  # 730048812
    },
    "serial": {
        "int_seq": ("all", INT),  # 1234
        "sn_prefix": ("all", STR),  # SN-0001234
        "gs1_21": ("all", STR),  # (21)7K3M9Q2X1B
        "uuid4": ("heldout", STR),  # 9f0c...-...
        "date_serial": ("heldout", STR),  # 24075-000123
    },
    "work_order": {
        "wo_year": ("all", STR),  # WO-2024-001523
        "fa_prefix": ("all", STR),  # FA4500123456
        "num10": ("all", STR),  # 1000234567
        "int10": ("heldout", INT),  # 1000234567
        "mo_slash": ("heldout", STR),  # MO/2024/01523
    },
    "sales_order": {
        "so_prefix": ("all", STR),  # SO-2024-004512
        "sap10": ("all", STR),  # 0012004512
        "kd_prefix": ("heldout", STR),  # KD-24-1187
    },
    "purchase_order": {
        "sap45": ("all", STR),  # 4500012345
        "po_prefix": ("all", STR),  # PO-102233
        "be_prefix": ("heldout", STR),  # BE24-00871
    },
    "tool": {
        "wz": ("all", STR),  # WZ-0815
        "mold": ("all", STR),  # M-12
        "tool_int": ("heldout", INT),  # 4711
        "fw_code": ("heldout", STR),  # FW-2210-03
    },
    "fixture": {
        "fx": ("all", STR),  # FX-03
        "vr": ("all", STR),  # VR-112
        "fix_alnum": ("heldout", STR),  # FIX_C07
    },
    "carrier": {
        "wt": ("all", STR),  # WT-017
        "epc_grai96": ("all", STR),  # 24 hex characters, GRAI-96
        "carrier_int": ("heldout", INT),  # 17
        "tray": ("heldout", STR),  # TR-0417
    },
    "machine": {
        "imm": ("all", STR),  # IMM-07
        "asset": ("all", STR),  # AS-10023
        "sap_equipment": ("heldout", STR),  # 10004711
        "line_machine": ("heldout", STR),  # L2M3
    },
    "station": {
        "st": ("all", STR),  # ST-040
        "op": ("all", STR),  # OP40
        "station_int": ("heldout", INT),  # 40
        "cell_station": ("heldout", STR),  # Z2-ST05
    },
    "location": {
        "aisle_rack_level": ("all", STR),  # A-03-12-2
        "rack_shelf_bin": ("all", STR),  # R12-S04-F2
        "compact_bin": ("heldout", STR),  # B0412C
    },
    "operator": {
        "badge_int": ("all", INT),  # 48213
        "p_number": ("all", STR),  # P12345
        "badge_padded": ("heldout", STR),  # 00048213
        "login": ("heldout", STR),  # kmeyer
    },
    "customer": {
        "c_number": ("all", STR),  # C100234
        "sap10": ("all", STR),  # 0000100234
        "k_dash": ("heldout", STR),  # K-4711
    },
    "supplier": {
        "v_dash": ("all", STR),  # V-20017
        "sap10": ("all", STR),  # 0000200017
        "sup_prefix": ("heldout", STR),  # SUP-0045
    },
    "handling_unit": {
        "sscc": ("all", STR),  # 18 digits, check digit
        "sscc_ai": ("all", STR),  # (00) + SSCC
        "sscc_int": ("heldout", INT),  # SSCC as an integer
        "hu_prefix": ("heldout", STR),  # HU-0004512
    },
}

#: Negative controls: ``{kind: (role, {variant: (split, value_column)})}``.
NEGATIVE_VARIANTS: dict[str, tuple[str, dict[str, tuple[str, str]]]] = {
    "machine_state": (
        "state",
        {
            "upper_en": ("all", STR),
            "title_en": ("all", STR),
            "german": ("heldout", STR),
        },
    ),
    "auto_manual": (
        "state",
        {"auto_manual": ("all", STR), "hand_auto": ("heldout", STR)},
    ),
    "step_name": (
        "state",
        {"en": ("all", STR), "numbered": ("all", STR), "de": ("heldout", STR)},
    ),
    "alarm_code": ("state", {"letter_digits": ("all", STR)}),  # hard: E1043, W0815
    "mode_code": ("state", {"int_123": ("all", INT)}),  # hard: low-cardinality code
    "shift_code": (
        "state",
        {
            "abc": ("all", STR),
            "early_late_night": ("all", STR),
            "shift_int": ("heldout", INT),
        },
    ),
    "quality_result": (
        "state",
        {"ok_nok": ("all", STR), "pass_fail": ("heldout", STR)},
    ),
    "part_counter": ("counter", {"int": ("all", INT)}),
    "energy_counter": ("counter", {"kwh": ("all", DBL)}),
    "temperature": ("measurement", {"float": ("all", DBL)}),
    "pressure_peak": ("measurement", {"float": ("all", DBL)}),
    "temperature_int": ("measurement", {"int": ("all", INT)}),
    "temperature_str": ("measurement", {"decimal_string": ("all", STR)}),  # "71.3"
    "shot_weight_int": ("measurement", {"int": ("all", INT)}),  # hard: big cardinality
    "alarm_text": ("free_text", {"en": ("all", STR), "de": ("heldout", STR)}),
    "alarm_code_text": ("free_text", {"coded": ("all", STR)}),  # hard: code tokens
    "mes_message": ("free_text", {"en": ("all", STR)}),  # hard: contains real ids
    "cycle_timestamp": (
        "timestamp",
        {
            "iso": ("all", STR),
            "minute": ("all", STR),
            "german": ("heldout", STR),
            "iso_ms_z": ("heldout", STR),
            "epoch_int": ("heldout", INT),
        },
    ),
    "production_date": (
        "timestamp",
        {"iso_date": ("all", STR), "compact_date": ("heldout", STR)},
    ),
    "firmware": ("constant", {"version": ("all", STR)}),
    "plant_code": ("constant", {"int": ("all", INT), "code": ("all", STR)}),
}

#: Labelled nesting: ``{child type: parent type}``.
NESTING: dict[str, str] = {"batch": "material", "serial": "batch"}

_LETTERS = np.array(list("ABCDEFGHJKLMNPRSTUVWXYZ"))
_ALNUM = np.array(list("0123456789ABCDEFGHJKLMNPRSTUVWXYZ"))
_HOUR = 3600.0
_MIN = 60.0
_DAY = 86400.0

Value: TypeAlias = object
Rng: TypeAlias = np.random.Generator


# -- small helpers -----------------------------------------------------------


def gs1_check_digit(data: str) -> str:
    """GS1 mod-10 check digit for the digits ``data`` (GTIN, SSCC, ...)."""
    total = sum(int(c) * (3 if i % 2 == 0 else 1) for i, c in enumerate(reversed(data)))
    return str((10 - total % 10) % 10)


def gs1_valid(code: str) -> bool:
    """True if the last digit of ``code`` is its correct GS1 check digit."""
    return code.isdigit() and len(code) > 1 and gs1_check_digit(code[:-1]) == code[-1]


def _letters(rng: Rng, n: int) -> str:
    return "".join(rng.choice(_LETTERS, n))


def _digits(rng: Rng, n: int) -> str:
    return "".join(str(d) for d in rng.integers(0, 10, n))


def _distinct(rng: Rng, n: int, make: Callable[[], Value]) -> list[Value]:
    seen: set[Value] = set()
    out: list[Value] = []
    while len(out) < n:
        value = make()
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _zipf_weights(rng: Rng, n: int, a: float) -> np.ndarray:
    w = 1.0 / np.arange(1, n + 1) ** a
    w = w[rng.permutation(n)]
    return np.asarray(w / w.sum())


def _duration(rng: Rng, mean: float, family: str) -> float:
    """One duration with the given mean, from a seed-chosen distribution."""
    if family == "lognormal":
        sigma = 0.5
        return float(rng.lognormal(np.log(mean) - sigma**2 / 2, sigma))
    if family == "exponential":
        return float(max(rng.exponential(mean), 0.1 * mean))
    if family == "gamma":
        return float(rng.gamma(4.0, mean / 4.0))
    return float(rng.uniform(0.5 * mean, 1.5 * mean))  # uniform


def _pick_variant(rng: Rng, variants: dict[str, tuple[str, str]], heldout: bool) -> str:
    names = [n for n, (split, _) in variants.items() if heldout or split == "all"]
    weights = np.array(
        [2.0 if variants[n][0] == "heldout" else 1.0 for n in names], dtype=float
    )
    return str(rng.choice(names, p=weights / weights.sum()))


def _abs_time(t0: pd.Timestamp, seconds: float) -> pd.Timestamp:
    return t0 + pd.Timedelta(seconds=float(seconds))


# -- identifier formats ------------------------------------------------------


def _pool(rng: Rng, object_type: str, variant: str, n: int, line: int) -> list[Value]:
    """``n`` distinct ids of a pooled type in the given format."""
    if object_type == "material":
        if variant == "sap18":
            nd = int(rng.integers(6, 9))
            return [
                f"{v:018d}"
                for v in _distinct(
                    rng, n, lambda: int(rng.integers(10 ** (nd - 1), 10**nd))
                )
            ]
        if variant == "mat_prefix":
            return [
                f"MAT-{v}"
                for v in _distinct(rng, n, lambda: int(rng.integers(100000, 999999)))
            ]
        if variant in ("gtin13", "gtin14"):
            prefix = str(rng.choice(["400", "401", "403", "405", "409", "426", "500"]))
            prefix += _digits(rng, int(rng.integers(4, 7)))
            width = 12 - len(prefix)
            items = _distinct(rng, n, lambda: _digits(rng, width))
            ind = str(rng.integers(1, 9))
            out: list[Value] = []
            for item in items:
                data = f"{prefix}{item}"
                if variant == "gtin14":
                    data = ind + data
                out.append(data + gs1_check_digit(data))
            return out
        if variant == "sap_int":
            return _distinct(rng, n, lambda: int(rng.integers(1_000_000, 99_999_999)))
        return _distinct(
            rng,
            n,
            lambda: f"{rng.integers(1, 10)}{_letters(rng, 2)}{_digits(rng, 4)}-"
            f"{rng.integers(0, 10)}{_letters(rng, 2)}",
        )
    if object_type == "recipe":
        if variant == "rz_name":
            polys = ["PA66", "PA6", "PP", "PBT", "POM", "ABS", "PC", "PPS", "TPE"]
            fills = ["GF30", "GF15", "GF50", "T20", "UNF", "MF40", "CF10", "GB25"]
            return _distinct(
                rng,
                n,
                lambda: f"RZ_{rng.choice(polys)}_{rng.choice(fills)}_v"
                f"{rng.integers(1, 10)}",
            )
        if variant == "prg":
            return [
                f"PRG{v:03d}"
                for v in _distinct(rng, n, lambda: int(rng.integers(1, 250)))
            ]
        if variant == "program_int":
            return _distinct(rng, n, lambda: int(rng.integers(4, 400)))
        products = ["HSG", "CVR", "BRK", "CLP", "FLG", "SHD", "CAP", "BSE"]
        colors = ["BK", "WH", "GY", "RD", "BL", "NT"]
        return _distinct(
            rng,
            n,
            lambda: f"{rng.choice(products)}-{rng.integers(100, 400)}-"
            f"{rng.choice(colors)}",
        )
    if object_type == "tool":
        if variant == "wz":
            return _distinct(rng, n, lambda: f"WZ-{rng.integers(100, 9999):04d}")
        if variant == "mold":
            return _distinct(rng, n, lambda: f"M-{rng.integers(1, 80)}")
        if variant == "tool_int":
            return _distinct(rng, n, lambda: int(rng.integers(1000, 9999)))
        return _distinct(
            rng, n, lambda: f"FW-{rng.integers(1000, 9999)}-{rng.integers(1, 9):02d}"
        )
    if object_type == "fixture":
        if variant == "fx":
            return _distinct(rng, n, lambda: f"FX-{rng.integers(1, 40):02d}")
        if variant == "vr":
            return _distinct(rng, n, lambda: f"VR-{rng.integers(100, 999)}")
        return _distinct(
            rng, n, lambda: f"FIX_{_letters(rng, 1)}{rng.integers(1, 99):02d}"
        )
    if object_type == "carrier":
        base = int(rng.integers(0, 3)) * 100
        if variant == "wt":
            return [f"WT-{base + k + 1:03d}" for k in range(n)]
        if variant == "carrier_int":
            return [base + k + 1 for k in range(n)]
        if variant == "tray":
            base = int(rng.integers(1, 9)) * 1000
            return [f"TR-{base + k + 1:04d}" for k in range(n)]
        company = int(rng.integers(1 << 20, 1 << 24))
        asset = int(rng.integers(1, 1 << 20))
        first = int(rng.integers(1, 1 << 20))
        out = []
        for k in range(n):
            word = (0x33 << 88) | (0 << 85) | (5 << 82) | (company << 58)
            word |= (asset << 38) | (first + k)
            out.append(f"{word:024X}")
        return out
    if object_type == "machine":
        if variant == "imm":
            first = int(rng.integers(1, 12))
            return [f"IMM-{first + k:02d}" for k in range(n)]
        if variant == "asset":
            return _distinct(rng, n, lambda: f"AS-{rng.integers(10000, 99999)}")
        if variant == "sap_equipment":
            return _distinct(rng, n, lambda: f"{rng.integers(10_000_000, 10_099_999)}")
        return [f"L{line + 1}M{k + 1}" for k in range(n)]
    if object_type == "station":
        if variant == "st":
            return [f"ST-{10 * (k + 1):03d}" for k in range(n)]
        if variant == "op":
            return [f"OP{10 * (k + 1)}" for k in range(n)]
        if variant == "station_int":
            return [10 * (k + 1) for k in range(n)]
        cell = int(rng.integers(1, 6))
        return [f"Z{cell}-ST{k + 1:02d}" for k in range(n)]
    if object_type == "location":
        if variant == "aisle_rack_level":
            return _distinct(
                rng,
                n,
                lambda: f"{_letters(rng, 1)}-{rng.integers(1, 20):02d}-"
                f"{rng.integers(1, 40):02d}-{rng.integers(1, 6)}",
            )
        if variant == "rack_shelf_bin":
            return _distinct(
                rng,
                n,
                lambda: f"R{rng.integers(1, 30):02d}-S{rng.integers(1, 12):02d}-"
                f"F{rng.integers(1, 6)}",
            )
        return _distinct(
            rng,
            n,
            lambda: f"{_letters(rng, 1)}{rng.integers(1, 30):02d}"
            f"{rng.integers(1, 40):02d}{_letters(rng, 1)}",
        )
    if object_type == "operator":
        if variant == "badge_int":
            nd = int(rng.integers(4, 7))
            return _distinct(rng, n, lambda: int(rng.integers(10 ** (nd - 1), 10**nd)))
        if variant == "p_number":
            return _distinct(rng, n, lambda: f"P{rng.integers(10000, 99999)}")
        if variant == "badge_padded":
            return _distinct(rng, n, lambda: f"{rng.integers(1000, 99999):08d}")
        initials = list("abcdefghjklmnprstw")
        last = [
            "meyer",
            "schmidt",
            "kaya",
            "novak",
            "berger",
            "fischer",
            "wagner",
            "becker",
            "hoffmann",
            "koch",
            "richter",
            "klein",
            "wolf",
            "neumann",
            "schulz",
            "braun",
            "zimmer",
            "kraus",
            "yilmaz",
            "lange",
        ]
        return _distinct(rng, n, lambda: f"{rng.choice(initials)}{rng.choice(last)}")
    if object_type == "customer":
        if variant == "c_number":
            return _distinct(rng, n, lambda: f"C{rng.integers(100000, 199999)}")
        if variant == "sap10":
            return _distinct(rng, n, lambda: f"{rng.integers(100000, 199999):010d}")
        return _distinct(rng, n, lambda: f"K-{rng.integers(1000, 9999)}")
    if object_type == "supplier":
        if variant == "v_dash":
            return _distinct(rng, n, lambda: f"V-{rng.integers(20000, 29999)}")
        if variant == "sap10":
            return _distinct(rng, n, lambda: f"{rng.integers(200000, 399999):010d}")
        return _distinct(rng, n, lambda: f"SUP-{rng.integers(1, 9999):04d}")
    raise AssertionError(object_type)  # pragma: no cover


def _numbers(rng: Rng, n: int, start: int, max_step: int) -> np.ndarray:
    """``n`` increasing numbers from ``start`` with random gaps."""
    return np.asarray(start + np.cumsum(rng.integers(1, max_step + 1, n)))


def _order_ids(
    rng: Rng, object_type: str, variant: str, n: int, year: int
) -> list[Value]:
    """``n`` increasing order numbers (work, sales or purchase orders)."""
    yy = year % 100
    if object_type == "work_order":
        if variant == "wo_year":
            return [f"WO-{year}-{v:06d}" for v in _numbers(rng, n, 1000, 25)]
        if variant == "fa_prefix":
            start = int(rng.integers(4_500_000_000, 4_600_000_000))
            return [f"FA{v:010d}" for v in _numbers(rng, n, start, 40)]
        if variant in ("num10", "int10"):
            start = int(rng.integers(1_000_000_000, 1_100_000_000))
            nums = _numbers(rng, n, start, 40)
            if variant == "int10":
                return [int(v) for v in nums]
            return [f"{v:010d}" for v in nums]
        return [f"MO/{year}/{v:05d}" for v in _numbers(rng, n, 100, 25)]
    if object_type == "sales_order":
        if variant == "so_prefix":
            return [f"SO-{year}-{v:06d}" for v in _numbers(rng, n, 2000, 30)]
        if variant == "sap10":
            start = int(rng.integers(10_000_000, 13_000_000))
            return [f"{v:010d}" for v in _numbers(rng, n, start, 30)]
        return [f"KD-{yy:02d}-{v:04d}" for v in _numbers(rng, n, 100, 20)]
    if object_type == "purchase_order":
        if variant == "sap45":
            start = int(rng.integers(4_500_000_000, 4_500_900_000))
            return [f"{v}" for v in _numbers(rng, n, start, 60)]
        if variant == "po_prefix":
            return [f"PO-{v:06d}" for v in _numbers(rng, n, 100000, 60)]
        return [f"BE{yy:02d}-{v:05d}" for v in _numbers(rng, n, 100, 40)]
    raise AssertionError(object_type)  # pragma: no cover


def _sscc_ids(rng: Rng, variant: str, n: int) -> list[Value]:
    ext = str(rng.integers(0, 10))
    company = str(rng.choice(["400", "401", "405", "426", "500", "760"]))
    company += _digits(rng, int(rng.integers(4, 7)))
    width = 16 - len(company)
    start = int(rng.integers(1, 10 ** (width - 1)))
    out: list[Value] = []
    for ref in _numbers(rng, n, start, 1 if rng.random() < 0.7 else 3):
        data = f"{ext}{company}{int(ref) % 10**width:0{width}d}"
        code = data + gs1_check_digit(data)
        if variant == "sscc":
            out.append(code)
        elif variant == "sscc_ai":
            out.append(f"(00){code}")
        elif variant == "sscc_int":
            out.append(int(code))
        else:
            out.append(f"HU-{int(ref) % 10**7:07d}")
    return out


def _batch_id(
    variant: str, ts: pd.Timestamp, seq_day: int, seq_week: int, line: int
) -> Value:
    iso = ts.isocalendar()
    if variant == "date_seq":
        return f"{ts:%y%m%d}-{seq_day:02d}"
    if variant == "week_seq":
        return f"B{iso[0] % 100:02d}W{iso[1]:02d}-{seq_week:03d}"
    if variant == "gs1_10":
        return f"(10){ts:%y%m%d}{'ABCD'[line]}{seq_day:02d}"
    if variant == "julian":
        return f"{ts:%y}{ts.dayofyear:03d}-{line + 1}-{seq_day:03d}"
    return int(f"{ts:%y%m%d}{seq_day:02d}")


def _lot_id(rng: Rng, variant: str, ts: pd.Timestamp) -> Value:
    if variant == "supplier_alnum":
        back = ts - pd.Timedelta(days=int(rng.integers(3, 90)))
        return f"L{back:%y%m}{_letters(rng, 1)}{rng.integers(1, 99):02d}"
    if variant == "charge":
        return f"CH-{rng.integers(10000, 99999)}-{rng.integers(1, 9)}"
    if variant == "gs1_10_lot":
        return f"(10){_letters(rng, 2)}{rng.integers(10000, 99999)}"
    return f"{rng.integers(100_000_000, 999_999_999)}"


def _serial_ids(
    rng: Rng, variant: str, starts_abs: pd.DatetimeIndex, first: int
) -> list[Value]:
    n = len(starts_abs)
    if variant == "int_seq":
        return [first + k for k in range(n)]
    if variant == "sn_prefix":
        return [f"SN-{first + k:07d}" for k in range(n)]
    if variant == "gs1_21":
        chars = rng.choice(_ALNUM, (n, 10))
        return ["(21)" + "".join(row) for row in chars]
    if variant == "uuid4":
        raw = rng.integers(0, 256, (n, 16), dtype=np.uint8)
        return [str(_uuid.UUID(bytes=bytes(row), version=4)) for row in raw]
    # date_serial: the sequence restarts every day, the date prefix keeps it unique
    days = starts_abs.normalize()
    seq = pd.Series(1, index=range(n)).groupby(np.asarray(days)).cumsum().to_numpy()
    yy = starts_abs.strftime("%y")
    doy = starts_abs.dayofyear
    return [f"{y}{d:03d}-{s:06d}" for y, d, s in zip(yy, doy, seq, strict=True)]


# -- simulation --------------------------------------------------------------


@dataclass
class _Params:
    heldout: bool
    family: str
    cycle_time: float
    wo_hours: float
    batch_hours: float
    run_minutes: float
    lot_hours: float
    n_material: int
    n_recipe: int
    n_tool: int
    n_fixture: int
    n_carrier: int
    n_machine: int
    n_station: int


def _params(rng: Rng, heldout: bool) -> _Params:
    if heldout:
        family = str(rng.choice(["lognormal", "exponential", "gamma", "uniform"]))
        return _Params(
            heldout=True,
            family=family,
            cycle_time=float(np.exp(rng.uniform(np.log(20), np.log(240)))),
            wo_hours=float(np.exp(rng.uniform(np.log(1.5), np.log(12)))),
            batch_hours=float(np.exp(rng.uniform(np.log(0.5), np.log(6)))),
            run_minutes=float(rng.uniform(15, 150)),
            lot_hours=float(rng.uniform(0.7, 10)),
            n_material=int(rng.integers(3, 41)),
            n_recipe=int(rng.integers(2, 10)),
            n_tool=int(rng.integers(2, 6)),
            n_fixture=int(rng.integers(2, 5)),
            n_carrier=int(rng.integers(20, 61)),
            n_machine=int(rng.integers(2, 7)),
            n_station=int(rng.integers(4, 14)),
        )
    return _Params(
        heldout=False,
        family="lognormal",
        cycle_time=float(rng.uniform(30, 70)),
        wo_hours=float(rng.uniform(3, 6)),
        batch_hours=float(rng.uniform(1.5, 3)),
        run_minutes=float(rng.uniform(40, 90)),
        lot_hours=float(rng.uniform(2, 5)),
        n_material=int(rng.integers(5, 21)),
        n_recipe=int(rng.integers(3, 7)),
        n_tool=int(rng.integers(2, 5)),
        n_fixture=int(rng.integers(2, 4)),
        n_carrier=int(rng.integers(25, 51)),
        n_machine=int(rng.integers(3, 6)),
        n_station=int(rng.integers(5, 11)),
    )


@dataclass
class _Line:
    index: int
    params: _Params
    # work-order entries (an interrupted order appears twice)
    entry_start: np.ndarray
    entry_setup_end: np.ndarray
    entry_end: np.ndarray
    entry_wo: list[int]
    entry_material: list[int]
    wo_count: int
    # machine state steps
    state_t: np.ndarray
    state_v: list[Value]
    run_end: np.ndarray  # every time the line leaves RUN (stop / setup)
    # cycles
    cycle_start: np.ndarray
    cycle_end: np.ndarray
    cycle_entry: np.ndarray
    # batch steps (key per run)
    batch_t: np.ndarray
    batch_key: list[int]
    batch_created: list[float]
    batch_seq_day: list[int]
    batch_seq_week: list[int]
    batch_entry: list[int]


def _simulate_line(
    rng: Rng, index: int, horizon: float, p: _Params, t0: pd.Timestamp
) -> _Line:
    weights = _zipf_weights(rng, p.n_material, 1.1)
    entries: list[tuple[float, float, float, int, int]] = []
    t = -float(rng.uniform(0.2, 0.9)) * p.wo_hours * _HOUR
    wo = 0
    prev_mat = -1

    def setup_len(same: bool) -> float:
        base = _duration(rng, 35 * _MIN, p.family)
        return float(np.clip(base * (0.35 if same else 1.0), 5 * _MIN, 3 * _HOUR))

    while t < horizon:
        same = prev_mat >= 0 and rng.random() < 0.35
        mat = prev_mat if same else int(rng.choice(p.n_material, p=weights))
        dur = max(_duration(rng, p.wo_hours * _HOUR, p.family), 0.4 * _HOUR)
        if rng.random() < 0.1 and dur > 2 * _HOUR:
            first = dur * float(rng.uniform(0.3, 0.6))
            s = setup_len(mat == prev_mat)
            entries.append((t, t + s, t + s + first, wo, mat))
            t += s + first
            rush_mat = int(rng.choice(p.n_material, p=weights))
            s = setup_len(rush_mat == mat)
            rush = float(rng.uniform(0.6, 1.8)) * _HOUR
            entries.append((t, t + s, t + s + rush, wo + 1, rush_mat))
            t += s + rush
            s = setup_len(rush_mat == mat)
            entries.append((t, t + s, t + s + dur - first, wo, mat))
            t += s + dur - first
            wo += 2
        else:
            s = setup_len(mat == prev_mat)
            entries.append((t, t + s, t + s + dur, wo, mat))
            t += s + dur
            wo += 1
        prev_mat = mat

    state_t: list[float] = []
    state_v: list[Value] = []
    run_end: list[float] = []
    c_start: list[np.ndarray] = []
    c_end: list[np.ndarray] = []
    c_entry: list[np.ndarray] = []
    ct = p.cycle_time
    for e, (start, setup_end, end, _wo, _mat) in enumerate(entries):
        state_t.append(start)
        state_v.append("SETUP")
        run_end.append(start)
        pt = setup_end
        while pt < end:
            run = max(_duration(rng, p.run_minutes * _MIN, p.family), 3 * ct)
            re = min(pt + run, end)
            state_t.append(pt)
            state_v.append("RUN")
            n_est = int((re - pt) / ct * 1.6) + 3
            d = ct * np.clip(1 + 0.07 * rng.standard_normal(n_est), 0.7, 1.8)
            st = pt + np.concatenate([[0.0], np.cumsum(d)[:-1]])
            keep = st + d <= re
            c_start.append(st[keep])
            c_end.append(st[keep] + 0.97 * d[keep])
            c_entry.append(np.full(int(keep.sum()), e))
            if re >= end:
                break
            idle = rng.random() < 0.6
            state_t.append(re)
            state_v.append("IDLE" if idle else "STOP")
            run_end.append(re)
            mean = (5 if idle else 22) * _MIN
            pt = min(re + _duration(rng, mean, p.family), end)

    cycle_start = np.concatenate(c_start) if c_start else np.empty(0)
    cycle_end = np.concatenate(c_end) if c_end else np.empty(0)
    cycle_entry = (
        np.concatenate(c_entry).astype(int) if c_entry else np.empty(0, dtype=int)
    )

    # Batches: a new batch at every order entry (an interrupted order usually
    # resumes its last batch), then a new one every ``batch_hours``, each
    # change snapped to the next cycle start.
    batch_t: list[float] = []
    batch_key: list[int] = []
    batch_created: list[float] = []
    batch_entry: list[int] = []
    last_batch_of_wo: dict[int, int] = {}
    for e, (start, setup_end, end, wo_id, _mat) in enumerate(entries):
        if wo_id in last_batch_of_wo and rng.random() < 0.7:
            key = last_batch_of_wo[wo_id]
        else:
            key = len(batch_created)
            batch_created.append(start)
        batch_t.append(start)
        batch_key.append(key)
        batch_entry.append(e)
        starts = cycle_start[cycle_entry == e]
        planned = setup_end + _duration(rng, p.batch_hours * _HOUR, p.family)
        while len(starts) and planned < end:
            j = int(np.searchsorted(starts, planned))
            if j >= len(starts):
                break
            key = len(batch_created)
            batch_created.append(float(starts[j]))
            batch_t.append(float(starts[j]))
            batch_key.append(key)
            batch_entry.append(e)
            planned = starts[j] + max(
                _duration(rng, p.batch_hours * _HOUR, p.family), 0.25 * _HOUR
            )
        last_batch_of_wo[wo_id] = key
    created = pd.DatetimeIndex([_abs_time(t0, c) for c in batch_created])
    seq_day = (
        pd.Series(1, index=range(len(created)))
        .groupby(np.asarray(created.normalize()))
        .cumsum()
    )
    week = [f"{d.isocalendar()[0]}-{d.isocalendar()[1]}" for d in created]
    seq_week = pd.Series(1, index=range(len(created))).groupby(week).cumsum()
    # the sequences of a week / day start mid-way at the recording start
    offset_day = int(rng.integers(0, 4))
    offset_week = int(rng.integers(0, 30))
    return _Line(
        index=index,
        params=p,
        entry_start=np.array([x[0] for x in entries]),
        entry_setup_end=np.array([x[1] for x in entries]),
        entry_end=np.array([x[2] for x in entries]),
        entry_wo=[x[3] for x in entries],
        entry_material=[x[4] for x in entries],
        wo_count=wo,
        state_t=np.array(state_t),
        state_v=state_v,
        run_end=np.array(run_end),
        cycle_start=cycle_start,
        cycle_end=cycle_end,
        cycle_entry=cycle_entry,
        batch_t=np.array(batch_t),
        batch_key=batch_key,
        batch_created=batch_created,
        batch_seq_day=[int(s) + offset_day for s in seq_day],
        batch_seq_week=[int(s) + offset_week for s in seq_week],
        batch_entry=batch_entry,
    )


# -- signals -----------------------------------------------------------------


@dataclass
class _Truth:
    """A signal before sampling: a step function or a list of stamped events."""

    key: str
    role: str
    object_type: str | None
    variant: str
    value_column: str
    times: np.ndarray
    values: list[Value]
    kind: str  # "held" | "event" | "series"
    parent: str | None = None
    hold_hint: float = 3600.0  # typical seconds a value is held


def _held(
    key: str,
    role: str,
    object_type: str | None,
    variant: str,
    column: str,
    times: Iterable[float],
    values: Iterable[Value],
    hold_hint: float,
    *,
    parent: str | None = None,
) -> _Truth:
    return _Truth(
        key,
        role,
        object_type,
        variant,
        column,
        np.asarray(list(times), dtype=float),
        list(values),
        "held",
        parent,
        hold_hint,
    )


def _steps_with_clears(
    times: np.ndarray,
    values: Sequence[Value],
    clear_times: np.ndarray,
    clear: Value,
) -> tuple[np.ndarray, list[Value]]:
    """Merge value steps with steps back to the cleared value."""
    all_t = np.concatenate([times, clear_times])
    all_v = list(values) + [clear] * len(clear_times)
    order = np.argsort(all_t, kind="stable")
    return all_t[order], [all_v[i] for i in order]


def _value_at(
    times: np.ndarray, values: Sequence[Value], at: np.ndarray
) -> list[Value]:
    idx = np.searchsorted(times, at, side="right") - 1
    return [values[i] if i >= 0 else None for i in idx]


def _line_signals(
    rng: Rng, line: _Line, t0: pd.Timestamp, horizon: float, heldout: bool
) -> list[_Truth]:
    p = line.params
    L = f"L{line.index}"
    out: list[_Truth] = []
    ct = p.cycle_time
    cs, ce = line.cycle_start, line.cycle_end

    def variant(object_type: str) -> tuple[str, str]:
        v = _pick_variant(rng, FORMAT_VARIANTS[object_type], heldout)
        return v, FORMAT_VARIANTS[object_type][v][1]

    def clear_value(column: str) -> Value:
        if column == INT:
            return 0 if rng.random() < 0.6 else None
        return "" if rng.random() < 0.7 else None

    # material / recipe / tool / fixture: attributes of the order's material
    mv, mcol = variant("material")
    materials = _pool(rng, "material", mv, p.n_material, line.index)
    out.append(
        _held(
            f"{L}.material",
            "identifier",
            "material",
            mv,
            mcol,
            line.entry_start,
            [materials[m] for m in line.entry_material],
            p.wo_hours * _HOUR,
        )
    )
    for otype, n, lag in (
        ("recipe", p.n_recipe, 0.0),
        ("tool", p.n_tool, 0.5),
        ("fixture", p.n_fixture, 0.6),
    ):
        v, col = variant(otype)
        ids = _pool(rng, otype, v, n, line.index)
        # contiguous groups of materials share a recipe / tool / fixture
        group = np.sort(rng.integers(0, n, p.n_material))
        group = group[rng.permutation(p.n_material)] if otype == "recipe" else group
        when = line.entry_start + lag * (line.entry_setup_end - line.entry_start)
        out.append(
            _held(
                f"{L}.{otype}",
                "identifier",
                otype,
                v,
                col,
                when,
                [ids[group[m]] for m in line.entry_material],
                p.wo_hours * _HOUR * 2,
            )
        )

    # work order
    wv, wcol = variant("work_order")
    year = int(_abs_time(t0, 0).year)
    orders = _order_ids(rng, "work_order", wv, line.wo_count, year)
    out.append(
        _held(
            f"{L}.work_order",
            "identifier",
            "work_order",
            wv,
            wcol,
            line.entry_start,
            [orders[w] for w in line.entry_wo],
            p.wo_hours * _HOUR,
        )
    )

    # batch, nested in the material
    bv, bcol = variant("batch")
    batch_ids = [
        _batch_id(
            bv,
            _abs_time(t0, line.batch_created[k]),
            line.batch_seq_day[k],
            line.batch_seq_week[k],
            line.index,
        )
        for k in range(len(line.batch_created))
    ]
    out.append(
        _held(
            f"{L}.batch",
            "identifier",
            "batch",
            bv,
            bcol,
            line.batch_t,
            [batch_ids[k] for k in line.batch_key],
            p.batch_hours * _HOUR,
            parent=f"{L}.material",
        )
    )

    # serial: one per cycle, cleared whenever the line stops; nested in batch
    sv, scol = variant("serial")
    starts_abs = pd.DatetimeIndex(t0 + pd.to_timedelta(cs, unit="s"))
    serials = _serial_ids(rng, sv, starts_abs, int(rng.integers(1, 2_000_000)))
    clear = clear_value(scol)
    st, svals = _steps_with_clears(cs, serials, line.run_end, clear)
    truth = _held(
        f"{L}.serial",
        "identifier",
        "serial",
        sv,
        scol,
        st,
        svals,
        ct,
        parent=f"{L}.batch",
    )
    out.append(truth)

    # companion part counter: +1 per finished part, reset per policy
    reset = str(rng.choice(["order", "shift", "day", "batch"]))
    count = np.ones(len(ce), dtype=np.int64)
    if reset == "order":
        group_id = line.cycle_entry
    elif reset == "batch":
        group_id = np.searchsorted(line.batch_t, cs, side="right")
    else:
        bounds = _shift_bounds(t0, horizon)
        if reset == "day":  # the production day starts at 06:00
            bounds = bounds[[_abs_time(t0, b).hour == 6 for b in bounds]]
        group_id = np.searchsorted(bounds, cs, side="right")
    counts = pd.Series(count).groupby(group_id).cumsum().to_numpy()
    out.append(
        _held(
            f"{L}.part_counter",
            "counter",
            None,
            f"part_counter:reset_{reset}",
            INT,
            np.concatenate([[0.0], ce]),
            [int(counts[0]) - 1, *[int(c) for c in counts]],
            ct,
        )
    )

    # carrier: a closed loop of N carriers past the reader, one per cycle
    cv, ccol = variant("carrier")
    carriers = _pool(rng, "carrier", cv, p.n_carrier, line.index)
    loop = list(range(p.n_carrier))
    out_of_loop: dict[int, float] = {}
    seq: list[Value] = []
    pos = int(rng.integers(0, p.n_carrier))
    for k in range(len(cs)):
        if rng.random() < 0.0015 and len(loop) > 10:  # pulled for cleaning
            c = loop.pop(int(rng.integers(0, len(loop))))
            out_of_loop[c] = cs[k] + float(rng.uniform(1, 6)) * _HOUR
            pos %= len(loop)
        for c, back in list(out_of_loop.items()):
            if cs[k] >= back:
                loop.insert(int(rng.integers(0, len(loop))), c)
                del out_of_loop[c]
        if rng.random() < 0.02:  # two carriers overtake each other
            j = (pos + 1) % len(loop)
            loop[pos], loop[j] = loop[j], loop[pos]
        seq.append(carriers[loop[pos]])
        pos = (pos + 1) % len(loop)
    clear = clear_value(ccol)
    if rng.random() < 0.5:
        ctimes, cvals = _steps_with_clears(cs, seq, ce, clear)
    else:
        ctimes, cvals = _steps_with_clears(cs, seq, line.run_end, clear)
    truth = _held(f"{L}.carrier", "identifier", "carrier", cv, ccol, ctimes, cvals, ct)
    out.append(truth)

    # machine: parallel machines, each part goes to the one that is free first
    mav, macol = variant("machine")
    machines = _pool(rng, "machine", mav, p.n_machine, line.index)
    free = np.zeros(p.n_machine)
    proc = p.n_machine * ct * 0.95
    down_m = int(rng.integers(0, p.n_machine))
    down_from = float(rng.uniform(0, horizon))
    down_to = down_from + float(rng.uniform(1, 6)) * _HOUR
    mseq: list[Value] = []
    for k in range(len(cs)):
        avail = free.copy()
        if down_from <= cs[k] < down_to and p.n_machine > 1:
            avail[down_m] = np.inf
        m = int(np.argmin(avail + rng.uniform(0, 0.3 * ct, p.n_machine)))
        free[m] = max(cs[k], free[m]) + proc * float(rng.uniform(0.85, 1.15))
        mseq.append(machines[m])
    out.append(_held(f"{L}.machine", "identifier", "machine", mav, macol, cs, mseq, ct))

    # station: a milk run visiting the stations in order
    sv2, stcol = variant("station")
    stations = _pool(rng, "station", sv2, p.n_station, line.index)
    tour_min = float(rng.uniform(20, 60) if not heldout else rng.uniform(10, 120))
    dwell_mean = tour_min * _MIN / p.n_station * 0.5
    stt: list[float] = []
    stv: list[Value] = []
    t = -float(rng.uniform(0, tour_min * _MIN))
    hold_to_next = rng.random() < 0.4
    clear = clear_value(stcol)
    while t < horizon:
        for s in stations:
            if rng.random() < 0.15:  # nothing to deliver there
                continue
            stt.append(t)
            stv.append(s)
            dwell = _duration(rng, dwell_mean, p.family)
            t += dwell
            if not hold_to_next:
                stt.append(t)
                stv.append(clear)
            t += _duration(rng, dwell_mean, p.family)
        t += float(rng.uniform(2, 10)) * _MIN  # back at the supermarket
    truth = _held(
        f"{L}.station",
        "identifier",
        "station",
        sv2,
        stcol,
        stt,
        stv,
        dwell_mean,
    )
    out.append(truth)

    # operator: logs in around the 06/14/22 shift changes
    ov, ocol = variant("operator")
    badges = _pool(rng, "operator", ov, int(rng.integers(6, 16)), line.index)
    crews = [badges[i::3] for i in range(3)]
    clear = clear_value(ocol)
    ot: list[float] = []
    oval: list[Value] = []
    bounds = _shift_bounds(t0, horizon)
    for b, nb in zip(bounds[:-1], bounds[1:], strict=True):
        slot = int(round(((_abs_time(t0, b).hour - 6) % 24) / 8)) % 3
        crew = crews[slot]
        who = crew[0] if rng.random() < 0.7 or len(crew) == 1 else crew[1]
        login = b + float(rng.uniform(-8, 20)) * _MIN
        ot.append(login)
        oval.append(who)
        if rng.random() < 0.3:  # relief over the break
            relief = badges[int(rng.integers(0, len(badges)))]
            rt = b + float(rng.uniform(3, 5)) * _HOUR
            ot += [rt, rt + float(rng.uniform(20, 40)) * _MIN]
            oval += [relief, who]
        logout = nb + float(rng.uniform(-15, 3)) * _MIN
        if rng.random() < 0.6:
            ot.append(logout)
            oval.append(clear)
    truth = _held(
        f"{L}.operator", "identifier", "operator", ov, ocol, ot, oval, 8 * _HOUR
    )
    out.append(truth)

    # lot: the raw material lot loaded on the line, swapped at each refill
    lv, lcol = variant("lot")
    lt: list[float] = []
    lval: list[Value] = []
    t = -float(rng.uniform(0, p.lot_hours * _HOUR))
    previous: Value = None
    current: Value = None
    while t < horizon:
        if previous is not None and rng.random() < 0.1:  # partial container back
            previous, current = current, previous
        else:
            previous, current = current, _lot_id(rng, lv, _abs_time(t0, max(t, 0)))
        lt.append(t)
        lval.append(current)
        t += max(_duration(rng, p.lot_hours * _HOUR, p.family), 0.3 * _HOUR)
    out.append(
        _held(f"{L}.lot", "identifier", "lot", lv, lcol, lt, lval, p.lot_hours * _HOUR)
    )

    out += _line_negatives(
        rng, line, t0, horizon, heldout, materials, orders, batch_ids
    )
    return out


def _shift_bounds(t0: pd.Timestamp, horizon: float) -> np.ndarray:
    midnight = t0.normalize()
    off = (t0 - midnight).total_seconds()
    days = int(horizon // _DAY) + 3
    b = [d * _DAY + h * _HOUR - off for d in range(-1, days) for h in (6.0, 14.0, 22.0)]
    arr = np.array(sorted(b))
    first = int(np.searchsorted(arr, 0.0, side="right")) - 1
    last = int(np.searchsorted(arr, horizon, side="left"))
    return np.asarray(arr[first : last + 1])


_STATE_NAMES = {
    "upper_en": {"RUN": "RUN", "IDLE": "IDLE", "STOP": "STOP", "SETUP": "SETUP"},
    "title_en": {"RUN": "Running", "IDLE": "Idle", "STOP": "Fault", "SETUP": "Setup"},
    "german": {
        "RUN": "PRODUKTION",
        "IDLE": "WARTEN",
        "STOP": "STOERUNG",
        "SETUP": "RUESTEN",
    },
}
_STEPS = {
    "numbered": ["S010", "S020", "S030", "S040", "S050", "S060"],
    "en": ["Close mold", "Inject", "Hold pressure", "Cool", "Open mold", "Eject"],
    "de": ["Schliessen", "Einspritzen", "Nachdruck", "Kuehlen", "Oeffnen", "Auswerfen"],
}
_ALARMS = {
    "en": [
        "E-Stop pressed at station {n}",
        "Safety door {n} open",
        "Hydraulic oil temperature high",
        "Low air pressure",
        "Material shortage at feeder {n}",
        "Part not ejected",
        "Cycle time exceeded",
        "Lubrication level low",
        "Barcode reader no read",
        "Light curtain {n} interrupted",
        "Gripper {n} vacuum not reached",
        "Conveyor jam detected",
    ],
    "de": [
        "Not-Halt betaetigt an Station {n}",
        "Schutztuer {n} offen",
        "Hydraulikoel Temperatur zu hoch",
        "Druckluft zu niedrig",
        "Materialmangel Zufuehrung {n}",
        "Teil nicht ausgeworfen",
        "Zykluszeit ueberschritten",
        "Schmierung Fuellstand niedrig",
        "Lichtgitter {n} unterbrochen",
    ],
}
_CODED = [
    "E{c}: Servo axis X{n} overtemp",
    "A-{c} Spindle drive fault FB-{n}",
    "[W{c}] Filter clogged, replace F-{n}",
    "ERR 0x{h} at module IM-{n}51",
    "F{c} PROFINET device IO-{n} lost",
    "Alarm {c} - Heater zone Z{n} open circuit",
]


def _line_negatives(
    rng: Rng,
    line: _Line,
    t0: pd.Timestamp,
    horizon: float,
    heldout: bool,
    materials: list[Value],
    orders: list[Value],
    batch_ids: list[Value],
) -> list[_Truth]:
    p = line.params
    L = f"L{line.index}"
    cs, ce = line.cycle_start, line.cycle_end
    ct = p.cycle_time
    out: list[_Truth] = []

    def variant(kind: str) -> tuple[str, str]:
        role, variants = NEGATIVE_VARIANTS[kind]
        v = _pick_variant(rng, variants, heldout)
        return v, variants[v][1]

    def neg(
        kind: str,
        v: str,
        col: str,
        times: Iterable[float],
        vals: Iterable[Value],
        hold: float,
        how: str = "held",
    ) -> _Truth:
        truth = _held(
            f"{L}.{kind}",
            NEGATIVE_VARIANTS[kind][0],
            None,
            f"{kind}:{v}",
            col,
            times,
            vals,
            hold,
        )
        truth.kind = how
        return truth

    v, col = variant("machine_state")
    names = _STATE_NAMES[v]
    out.append(
        neg(
            "machine_state",
            v,
            col,
            line.state_t,
            [names[str(s)] for s in line.state_v],
            p.run_minutes * _MIN,
        )
    )

    manual = [
        s == "SETUP" or (s == "STOP" and rng.random() < 0.5) for s in line.state_v
    ]
    v, col = variant("auto_manual")
    words = ("AUTO", "MANUAL") if v == "auto_manual" else ("AUTOMATIK", "HAND")
    out.append(
        neg(
            "auto_manual",
            v,
            col,
            line.state_t,
            [words[int(m)] for m in manual],
            p.run_minutes * _MIN,
        )
    )
    codes = [
        3 if s == "SETUP" else (2 if m else 1)
        for s, m in zip(line.state_v, manual, strict=True)
    ]
    out.append(
        neg("mode_code", "int_123", INT, line.state_t, codes, p.run_minutes * _MIN)
    )

    if ct >= 40 and line.index == 0 and rng.random() < 0.7:
        v, col = variant("step_name")
        steps = _STEPS[v]
        frac = np.cumsum([0, 0.08, 0.17, 0.25, 0.32, 0.08])
        times = (cs[:, None] + frac[None, :] * (ce - cs)[:, None]).ravel()
        wait = {"en": "Wait", "de": "Warten", "numbered": "S000"}[v]
        step_t, step_v = _steps_with_clears(times, steps * len(cs), ce, wait)
        out.append(neg("step_name", v, col, step_t, step_v, ct / 6))

    v, col = variant("shift_code")
    bounds = _shift_bounds(t0, horizon)
    shift_names: dict[str, Sequence[Value]] = {
        "abc": "ABC",
        "early_late_night": ["Early", "Late", "Night"],
        "shift_int": [1, 2, 3],
    }
    labels = shift_names[v]
    shift_vals = [
        labels[int(round(((_abs_time(t0, b).hour - 6) % 24) / 8)) % 3]
        for b in bounds[:-1]
    ]
    out.append(neg("shift_code", v, col, bounds[:-1], shift_vals, 8 * _HOUR))

    if rng.random() < 0.7:
        v, col = variant("quality_result")
        good, bad = ("OK", "NOK") if v == "ok_nok" else ("PASS", "FAIL")
        vals = [bad if r < 0.03 else good for r in rng.random(len(ce))]
        out.append(neg("quality_result", v, col, ce, vals, ct, "event"))

    # energy counter (kWh, monotonic)
    if rng.random() < 0.6:
        period = float(rng.choice([60.0, 300.0, 900.0]))
        grid = np.arange(0.0, horizon, period)
        running = np.isin(
            np.asarray(_value_at(line.state_t, line.state_v, grid), dtype=object),
            ["RUN"],
        )
        kw = np.where(running, rng.uniform(40, 120), rng.uniform(5, 15))
        kwh = float(rng.uniform(1e4, 9e5)) + np.cumsum(kw * period / _HOUR)
        out.append(
            neg(
                "energy_counter",
                "kwh",
                DBL,
                grid,
                np.round(kwh, 3).tolist(),
                period,
                "series",
            )
        )

    # temperature follows a recipe-dependent setpoint
    period = float(rng.choice([15.0, 30.0, 60.0]))
    grid = np.arange(float(rng.uniform(0, period)), horizon, period)
    entry_now = np.searchsorted(line.entry_start, grid, side="right") - 1
    setpoints = rng.uniform(40, 95, p.n_material)
    target = setpoints[np.asarray(line.entry_material)[np.maximum(entry_now, 0)]]
    temp = np.empty(len(grid))
    x = target[0] if len(grid) else 0.0
    noise = rng.standard_normal(len(grid))
    for i in range(len(grid)):
        x += 0.05 * (target[i] - x) + 0.15 * noise[i]
        temp[i] = x
    out.append(
        neg(
            "temperature",
            "float",
            DBL,
            grid,
            np.round(temp, 2).tolist(),
            period,
            "series",
        )
    )
    if rng.random() < 0.8:
        step = max(1, int(round(60 / period)))
        out.append(
            neg(
                "temperature_int",
                "int",
                INT,
                grid[::step],
                [int(round(t)) for t in temp[::step]],
                period,
                "series",
            )
        )

    if rng.random() < 0.5:  # an OPC string tag holding a number
        step = max(1, int(round(60 / period)))
        out.append(
            neg(
                "temperature_str",
                "decimal_string",
                STR,
                grid[::step],
                [f"{t:.1f}" for t in temp[::step]],
                period,
                "series",
            )
        )

    peak = rng.uniform(600, 1400) + 15 * rng.standard_normal(len(ce))
    out.append(
        neg("pressure_peak", "float", DBL, ce, np.round(peak, 1).tolist(), ct, "event")
    )
    weight = rng.uniform(5000, 40000) * (1 + 0.004 * rng.standard_normal(len(ce)))
    out.append(
        neg("shot_weight_int", "int", INT, ce, [int(w) for w in weight], ct, "event")
    )

    # alarms: at every stop plus a few warnings
    stops = np.array(
        [t for t, s in zip(line.state_t, line.state_v, strict=True) if s == "STOP"]
    )
    extra = np.sort(rng.uniform(0, horizon, int(rng.poisson(horizon / (3 * _HOUR)))))
    at = np.sort(np.concatenate([stops, extra]))
    v, col = variant("alarm_text")
    templates = _ALARMS[v]
    texts = [
        templates[int(rng.integers(0, len(templates)))].format(n=rng.integers(1, 9))
        for _ in at
    ]
    clear_after = rng.random() < 0.5
    alarm_t: np.ndarray = at
    alarm_v: list[Value] = list(texts)
    if clear_after:
        alarm_t, alarm_v = _steps_with_clears(at, texts, at + rng.uniform(20, 600), "")
    out.append(neg("alarm_text", v, col, alarm_t, alarm_v, 300.0, "event"))

    if rng.random() < 0.5:  # the code of the active fault, 0-like when none
        pool = [
            f"{rng.choice(list('EWAF'))}{rng.integers(100, 9999):04d}"
            for _ in range(int(rng.integers(6, 25)))
        ]
        pw = _zipf_weights(rng, len(pool), 1.2)
        codes_at = [pool[int(i)] for i in rng.choice(len(pool), len(at), p=pw)]
        ctimes, cvals = _steps_with_clears(
            at, codes_at, at + rng.uniform(30, 900, len(at)), ""
        )
        out.append(neg("alarm_code", "letter_digits", STR, ctimes, cvals, 300.0))

    if rng.random() < 0.85:
        at2 = np.sort(rng.uniform(0, horizon, int(rng.poisson(horizon / _HOUR))))
        coded = [
            _CODED[int(rng.integers(0, len(_CODED)))].format(
                c=rng.integers(1000, 9999),
                n=rng.integers(1, 9),
                h=f"{rng.integers(0, 65535):04X}",
            )
            for _ in at2
        ]
        out.append(neg("alarm_code_text", "coded", STR, at2, coded, 300.0, "event"))

    if rng.random() < 0.85:
        msgs: list[Value] = []
        mt: list[float] = []
        for e in range(len(line.entry_start)):
            mt.append(float(line.entry_start[e]))
            msgs.append(
                f"Order {orders[line.entry_wo[e]]} started, material "
                f"{materials[line.entry_material[e]]}"
            )
        for t, k in zip(line.batch_t, line.batch_key, strict=True):
            mt.append(float(t) + 1.0)
            msgs.append(f"Batch {batch_ids[k]} released")
        order = np.argsort(mt)
        out.append(
            neg(
                "mes_message",
                "en",
                STR,
                np.asarray(mt)[order],
                [msgs[i] for i in order],
                3600.0,
                "event",
            )
        )

    v, col = variant("cycle_timestamp")
    stamp_abs = pd.DatetimeIndex(t0 + pd.to_timedelta(np.floor(cs), unit="s"))
    if v == "epoch_int":
        stamps: list[Value] = [int(x) for x in stamp_abs.asi8 // 10**9]
    else:
        fmt = {
            "iso": "%Y-%m-%dT%H:%M:%S",
            "minute": "%Y-%m-%d %H:%M",
            "german": "%d.%m.%Y %H:%M:%S",
            "iso_ms_z": "%Y-%m-%dT%H:%M:%S.000Z",
        }[v]
        stamps = list(stamp_abs.strftime(fmt))
    out.append(neg("cycle_timestamp", v, col, cs, stamps, ct))

    if rng.random() < 0.6:
        v, col = variant("production_date")
        # the production day starts with the early shift at 06:00
        days = pd.date_range(
            (t0 - pd.Timedelta(hours=6)).normalize(), periods=int(horizon // _DAY) + 2
        )
        dt = [(d + pd.Timedelta(hours=6) - t0).total_seconds() for d in days]
        fmt = "%Y-%m-%d" if v == "iso_date" else "%Y%m%d"
        out.append(neg("production_date", v, col, dt, list(days.strftime(fmt)), _DAY))

    v, col = variant("firmware")
    fw = f"V{rng.integers(1, 5)}.{rng.integers(0, 20)}.{rng.integers(0, 9)}"
    out.append(neg("firmware", v, col, [0.0], [fw], horizon))
    if rng.random() < 0.5:
        v, col = variant("plant_code")
        code: Value = int(rng.choice([1000, 1100, 2000, 4711]))
        if col == STR:
            code = f"DE{rng.integers(1, 20):02d}"
        out.append(neg("plant_code", v, col, [0.0], [code], horizon))
    return out


def _plant_signals(
    rng: Rng, line: _Line, t0: pd.Timestamp, horizon: float, heldout: bool
) -> list[_Truth]:
    """Shipping, goods receipt and warehouse signals of the plant."""
    out: list[_Truth] = []
    year = int(_abs_time(t0, 0).year)

    def variant(object_type: str) -> tuple[str, str]:
        v = _pick_variant(rng, FORMAT_VARIANTS[object_type], heldout)
        return v, FORMAT_VARIANTS[object_type][v][1]

    def stamped(
        order_type: str, partner_type: str, mean_gap: float, repeat: float
    ) -> None:
        events = []
        t = float(rng.uniform(0, mean_gap))
        while t < horizon:
            events.append(t)
            t += _duration(rng, mean_gap, "exponential")
        n = len(events)
        ov, ocol = variant(order_type)
        pv, pcol = variant(partner_type)
        ids = _order_ids(rng, order_type, ov, n, year)
        partners = _pool(rng, partner_type, pv, int(rng.integers(5, 26)), 0)
        pw = _zipf_weights(rng, len(partners), 1.0)
        k = -1
        seq: list[int] = []
        for _ in range(n):  # an order is stamped again for each shipment / line
            if k < 0 or rng.random() > repeat:
                k += 1
            seq.append(k)
        partner_of = rng.choice(len(partners), size=max(n, 1), p=pw)
        ev = np.asarray(events)
        lag = 0.0 if rng.random() < 0.5 else float(rng.uniform(0.2, 3))
        out.append(
            _Truth(
                f"P.{order_type}",
                "identifier",
                order_type,
                ov,
                ocol,
                ev,
                [ids[i] for i in seq],
                "event",
            )
        )
        out.append(
            _Truth(
                f"P.{partner_type}",
                "identifier",
                partner_type,
                pv,
                pcol,
                ev + lag,
                [partners[int(partner_of[i])] for i in seq],
                "event",
            )
        )

    scale = float(rng.uniform(0.5, 2.5)) if heldout else 1.0
    stamped("sales_order", "customer", 80 * _MIN * scale, 0.35)
    stamped("purchase_order", "supplier", 120 * _MIN * scale, 0.25)

    # handling units: a box of n parts on the first line, SSCC per box
    ce = line.cycle_end
    size = int(rng.integers(20, 201))
    n_hu = len(ce) // size + 1
    hv, hcol = variant("handling_unit")
    hus = _sscc_ids(rng, hv, n_hu)
    first_part = line.cycle_start[::size]
    closed = ce[size - 1 :: size]
    if rng.random() < 0.5:
        out.append(
            _held(
                "P.handling_unit",
                "identifier",
                "handling_unit",
                hv,
                hcol,
                first_part,
                hus[: len(first_part)],
                size * line.params.cycle_time,
            )
        )
    else:
        out.append(
            _Truth(
                "P.handling_unit",
                "identifier",
                "handling_unit",
                hv,
                hcol,
                closed,
                hus[: len(closed)],
                "event",
            )
        )

    # put-away location of each closed box
    lv, lcol = variant("location")
    bins = _pool(rng, "location", lv, int(rng.integers(80, 400)), 0)
    put = np.sort(closed + rng.uniform(3, 30, len(closed)) * _MIN)
    put = put[put < horizon]
    chosen = rng.integers(0, len(bins), len(put))
    out.append(
        _Truth(
            "P.location",
            "identifier",
            "location",
            lv,
            lcol,
            put,
            [bins[int(c)] for c in chosen],
            "event",
        )
    )
    return out


# -- sampling and noise ------------------------------------------------------


def _gaps(rng: Rng, horizon: float, rate_per_day: float) -> list[tuple[float, float]]:
    n = int(rng.poisson(rate_per_day * horizon / _DAY))
    out = []
    for _ in range(n):
        s = float(rng.uniform(0, horizon))
        d = float(np.exp(rng.uniform(np.log(5 * _MIN), np.log(2 * _HOUR))))
        out.append((s, min(s + d, horizon)))
    return out


def _outside(t: np.ndarray, gaps: list[tuple[float, float]]) -> np.ndarray:
    keep = np.ones(len(t), dtype=bool)
    for s, e in gaps:
        keep &= ~((t >= s) & (t < e))
    return keep


def _sample(
    rng: Rng, truth: _Truth, horizon: float, gaps: list[tuple[float, float]]
) -> tuple[np.ndarray, list[Value], bool]:
    """Rows a historian would store for ``truth``: ``(times, values, is_delta)``."""
    t = truth.times
    vals = truth.values
    if truth.kind in ("event", "series"):
        keep = (t >= 0) & (t < horizon) & _outside(t, gaps)
        idx = np.flatnonzero(keep)
        return t[idx], [vals[i] for i in idx], truth.kind == "event"

    # held: clip to the horizon (the value at 0 is the last one set before it)
    order = np.argsort(t, kind="stable")
    t = t[order]
    vals = [vals[i] for i in order]
    first = max(int(np.searchsorted(t, 0.0, side="right")) - 1, 0)
    t = np.maximum(t[first:], 0.0)
    vals = vals[first:]
    inside = t < horizon
    t = t[inside]
    vals = [v for v, k in zip(vals, inside, strict=True) if k]
    # collapse repeats and same-time overwrites to the last value
    if len(t) > 1:
        last_at_time = np.r_[t[1:] != t[:-1], True]
        t = t[last_at_time]
        vals = [v for v, k in zip(vals, last_at_time, strict=True) if k]
        change = np.r_[True, [vals[i] != vals[i - 1] for i in range(1, len(vals))]]
        t = t[change]
        vals = [v for v, k in zip(vals, change, strict=True) if k]

    periodic = rng.random() < 0.4 and truth.hold_hint >= 30
    if periodic:
        if truth.hold_hint < 600:
            period = float(np.clip(truth.hold_hint / 2, 10, 60))
        else:
            period = float(rng.choice([30.0, 60.0, 120.0, 300.0]))
        grid = np.arange(float(rng.uniform(0, period)), horizon, period)
        grid = grid + rng.uniform(-0.02, 0.02, len(grid)) * period
        grid = grid[(grid >= 0) & _outside(grid, gaps)]
        return grid, _value_at(t, vals, grid), False

    rows = [t]
    if truth.hold_hint >= 600 and rng.random() < 0.5:  # keep-alive re-sends
        beat = float(rng.choice([300.0, 600.0, 900.0, 3600.0]))
        rows.append(np.arange(float(rng.uniform(0, beat)), horizon, beat))
    rows.append(np.array([e for _, e in gaps if e < horizon]))  # reconnect
    all_t = np.unique(np.concatenate(rows))
    all_t = all_t[_outside(all_t, gaps)]
    return all_t, _value_at(t, vals, all_t), True


_GARBAGE = ["NOREAD", "????????", "ERR", "#N/A", "0", "--------", "NULL"]


def _noisy(rng: Rng, truth: _Truth, values: list[Value], heldout: bool) -> list[Value]:
    level = int(rng.choice(3, p=[0.25, 0.45, 0.3] if heldout else [0.35, 0.45, 0.2]))
    if level == 0:
        return values
    n = len(values)
    out = list(values)
    na_rate = (0.0, 0.003, 0.012)[level]
    for i in np.flatnonzero(rng.random(n) < na_rate):
        out[i] = None
    if truth.role not in ("identifier", "state"):
        return out
    junk = (0.0, 0.002, 0.008)[level]
    junk_idx = np.flatnonzero(rng.random(n) < junk)
    if truth.value_column == INT:
        for i in junk_idx:
            out[i] = int(rng.choice([0, -1, 32767, 65535]))
        return out
    for i in junk_idx:
        v = out[i]
        if isinstance(v, str) and len(v) > 3 and rng.random() < 0.4:
            out[i] = v[: int(rng.integers(1, len(v)))]  # truncated read
        else:
            out[i] = str(rng.choice(_GARBAGE))
    for i in np.flatnonzero(rng.random(n) < (0.0, 0.003, 0.01)[level]):
        out[i] = ""
    if rng.random() < 0.3:  # typed in by hand now and then
        for i in np.flatnonzero(rng.random(n) < 0.02):
            v = out[i]
            if isinstance(v, str):
                out[i] = v.lower() if rng.random() < 0.7 else v.swapcase()
    if rng.random() < 0.3:  # a second system strips the leading zeros
        for i in np.flatnonzero(rng.random(n) < 0.05):
            v = out[i]
            if isinstance(v, str) and v.isdigit() and v.startswith("0"):
                out[i] = v.lstrip("0") or "0"
    return out


# -- entry point -------------------------------------------------------------


def _selected(include: Iterable[str] | None) -> set[str] | None:
    if include is None:
        return None
    names = {include} if isinstance(include, str) else set(include)
    known = set(OBJECT_TYPES) | set(ROLES) | set(NEGATIVE_VARIANTS)
    unknown = sorted(names - known)
    if unknown:
        raise ValueError(
            f"include names unknown object types or roles: {unknown}; use object "
            f"types {list(OBJECT_TYPES)}, roles {list(ROLES)} or negative kinds "
            f"{list(NEGATIVE_VARIANTS)}"
        )
    return names


def generate(
    *,
    seed: int,
    days: float,
    include: Iterable[str] | None,
    noise: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if days <= 0:
        raise ValueError(f"days must be positive; got {days}")
    wanted = _selected(include)
    rng = np.random.default_rng(seed)
    heldout = seed >= HELDOUT_SEED_MIN
    horizon = float(days) * _DAY
    t0 = pd.Timestamp("2024-01-01") + pd.Timedelta(
        days=int(rng.integers(0, 730)), hours=int(rng.integers(0, 24))
    )
    n_lines = int(rng.choice([1, 2], p=[0.4, 0.6] if heldout else [0.5, 0.5]))
    truths: list[_Truth] = []
    line_gaps: dict[str, list[tuple[float, float]]] = {}
    lines = []
    params = [_params(rng, heldout) for _ in range(n_lines)]
    # keep the frame at a laptop-friendly size: drop the second line when
    # both together would make too many cycles
    if n_lines == 2 and sum(horizon / p.cycle_time for p in params) > 4500 * days:
        params = params[:1]
    for i, p in enumerate(params):
        line = _simulate_line(rng, i, horizon, p, t0)
        lines.append(line)
        sigs = _line_signals(rng, line, t0, horizon, heldout)
        truths += sigs
        line_gaps[f"L{i}"] = _gaps(rng, horizon, 0.7)
    truths += _plant_signals(rng, lines[0], t0, horizon, heldout)
    line_gaps["P"] = _gaps(rng, horizon, 0.3)

    if wanted is not None:
        truths = [
            s
            for s in truths
            if s.object_type in wanted
            or s.role in wanted
            or s.variant.split(":")[0] in wanted
        ]
    order = rng.permutation(len(truths))
    uuid_of = {truths[j].key: f"sig:{k:04d}" for k, j in enumerate(order)}

    frames = []
    labels = []
    for truth in truths:
        gaps = list(line_gaps[truth.key.split(".")[0]])
        if noise and rng.random() < 0.3:
            gaps += _gaps(rng, horizon, 0.5)
        times, values, is_delta = _sample(rng, truth, horizon, gaps)
        if noise:
            values = _noisy(rng, truth, values, heldout)
        if all(v is None for v in values):  # nothing recorded (a very short run)
            continue
        uid = uuid_of[truth.key]
        frames.append(_frame(t0, uid, truth.value_column, times, values, is_delta))
        labels.append(
            {
                "uuid": uid,
                "role": truth.role,
                "object_type": truth.object_type,
                "value_column": truth.value_column,
                "parent_uuid": uuid_of.get(truth.parent) if truth.parent else None,
                "variant": truth.variant,
            }
        )
    present = {row["uuid"] for row in labels}
    for row in labels:
        if row["parent_uuid"] not in present:
            row["parent_uuid"] = None
    columns = ["uuid", "role", "object_type", "value_column", "parent_uuid", "variant"]
    label_frame = (
        pd.DataFrame(labels, columns=columns)
        .sort_values("uuid", ignore_index=True)
        .astype(object)
    )
    label_frame = label_frame.where(label_frame.notna(), None)
    if not frames:
        return empty_signal_frame(), label_frame
    frame = pd.concat(frames, ignore_index=True)
    frame = frame.sort_values(["uuid", "systime"], kind="stable", ignore_index=True)
    return frame, label_frame


def _frame(
    t0: pd.Timestamp,
    uid: str,
    column: str,
    times: np.ndarray,
    values: list[Value],
    is_delta: bool,
) -> pd.DataFrame:
    n = len(times)
    ms = np.round(np.asarray(times, dtype=float) * 1000).astype("int64")
    systime = (t0 + pd.to_timedelta(ms, unit="ms")).astype("datetime64[ns]")
    integers = pd.array([pd.NA] * n, dtype="Int64")
    doubles = np.full(n, np.nan)
    strings = np.full(n, None, dtype=object)
    if column == INT:
        integers = pd.array(values, dtype="Int64")
    elif column == DBL:
        doubles = np.array([np.nan if v is None else v for v in values], dtype=float)
    else:
        strings = np.array(values, dtype=object)
    return pd.DataFrame(
        {
            "systime": systime,
            "uuid": uid,
            "value_bool": pd.array([pd.NA] * n, dtype="boolean"),
            "value_integer": integers,
            "value_double": doubles,
            "value_string": strings,
            "is_delta": is_delta,
        }
    )
