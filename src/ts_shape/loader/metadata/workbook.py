"""Read a plant's datapoint workbook (.xlsx) without extra dependencies.

A datapoint export is often delivered as a workbook: one row per datapoint
with ``uuid``, ``label`` and the flattened configuration as ``config.<key>``
columns (``config.nodePath[string]``, ``config.unit``, ...), next to plain
columns such as ``is_enabled``, ``is_archived``, ``scaling_factor`` and
``scaling_offset``. It is the flat form of the JSON that
:class:`~ts_shape.loader.metadata.metadata_json_loader.MetadataJsonLoader`
reads.

* :func:`read_workbook` reads one sheet as a frame (header in the first row)
  with the Python standard library only (``zipfile`` + XML), so no Excel
  package is needed;
* :func:`nested_records` turns such a flat frame into the records with a
  nested ``config`` dict the JSON loader takes.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import pandas as pd

__all__ = ["nested_records", "read_workbook"]

_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
_REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
_TYPE_SUFFIX = re.compile(r"\[[^\]]*\]$")


def _column_index(ref: str) -> int:
    match = re.match(r"[A-Z]+", ref)
    letters = match.group(0) if match else "A"
    index = 0
    for ch in letters:
        index = index * 26 + ord(ch) - 64
    return index - 1


def read_workbook(path: str | Path, sheet: int | str = 0) -> pd.DataFrame:
    """One sheet of an ``.xlsx`` as a frame (header in row 1).

    Args:
        path: The workbook file.
        sheet: Sheet position (0 = first) or name.

    Returns:
        One row per non-empty sheet row; text, numbers (integral numbers as
        ``int``) and booleans as stored. Empty cells are missing.

    Raises:
        FileNotFoundError: When ``path`` does not exist.
        ValueError: When the file is not a workbook or has no such sheet.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"read_workbook: no file {path}")
    try:
        book = zipfile.ZipFile(path)
    except zipfile.BadZipFile:
        raise ValueError(f"read_workbook: {path} is not an .xlsx workbook") from None
    with book:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in book.namelist():
            root = ET.fromstring(book.read("xl/sharedStrings.xml"))
            for item in root.findall("m:si", _NS):
                shared.append(
                    "".join(t.text or "" for t in item.iter(f"{{{_NS['m']}}}t"))
                )
        workbook = ET.fromstring(book.read("xl/workbook.xml"))
        sheets = workbook.findall("m:sheets/m:sheet", _NS)
        if isinstance(sheet, int):
            chosen = sheets[sheet] if -len(sheets) <= sheet < len(sheets) else None
        else:
            chosen = next((s for s in sheets if s.get("name") == sheet), None)
        if chosen is None:
            names = [s.get("name") for s in sheets]
            raise ValueError(f"read_workbook: no sheet {sheet!r} in {path}: {names}")
        rels = ET.fromstring(book.read("xl/_rels/workbook.xml.rels"))
        target = next(r.get("Target") for r in rels if r.get("Id") == chosen.get(_REL))
        assert target is not None
        xml = ET.fromstring(book.read("xl/" + target.lstrip("/").removeprefix("xl/")))

    rows: list[dict[int, Any]] = []
    for row in xml.iter(f"{{{_NS['m']}}}row"):
        values: dict[int, Any] = {}
        for cell in row.findall("m:c", _NS):
            kind = cell.get("t")
            v = cell.find("m:v", _NS)
            value: Any
            if kind == "inlineStr":
                value = "".join(t.text or "" for t in cell.iter(f"{{{_NS['m']}}}t"))
            elif v is None or v.text is None:
                continue
            elif kind == "s":
                value = shared[int(v.text)]
            elif kind == "b":
                value = v.text == "1"
            elif kind in ("str", "e"):
                value = v.text
            else:
                number = float(v.text)
                value = int(number) if number.is_integer() else number
            if value == "":
                continue
            values[_column_index(cell.get("r") or "A")] = value
        rows.append(values)
    if not rows:
        return pd.DataFrame()
    header = rows[0]
    order = sorted(header)
    names = [str(header[i]) for i in order]
    data = [[r.get(i) for i in order] for r in rows[1:] if r]
    return pd.DataFrame(data, columns=names)


def _missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def nested_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Records with a nested ``config`` from a flat datapoint table.

    ``config.<key>`` and ``config.<key>[type]`` columns become keys of the
    record's ``config`` dict (the ``[type]`` suffix of an export is
    dropped); every other column stays at the top level. Missing values are
    left out.
    """
    records: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        record: dict[str, Any] = {}
        config: dict[str, Any] = {}
        for column, value in row.items():
            if _missing(value):
                continue
            name = str(column)
            if name.startswith("config."):
                key = _TYPE_SUFFIX.sub("", name[len("config.") :])
                config.setdefault(key, value)
            else:
                record[name] = value
        record["config"] = config
        if "uuid" in record:
            record["uuid"] = str(record["uuid"])
        records.append(record)
    return records
