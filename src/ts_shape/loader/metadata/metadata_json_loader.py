import logging
import json
from typing import Dict, Any, List, Optional, Iterable
import pandas as pd

logger = logging.getLogger(__name__)


class MetadataJsonLoader:
    """
    Load metadata JSON of shape:
      {
        "uuid":   {"0": "...", "1": "...", ...},
        "label":  {"0": "...", "1": "...", ...},
        "config": {"0": {...},  "1": {...},  ...}
      }
    into a pandas DataFrame with flattened config columns.
    """

    def __init__(self, json_data: Any, *, strict: bool = True):
        """
        Create a loader from JSON-like data.

        Args:
            json_data: Supported shapes:
              - dict of columns with index-maps: {"uuid": {"0": ...}, "label": {...}, "config": {...}}
              - dict of columns with lists: {"uuid": [...], "label": [...], "config": [...]}
              - list of records: [{"uuid": ..., "label": ..., "config": {...}}, ...]
            strict: If True, enforce presence of required keys and unique UUIDs.
        """
        self.json_data = json_data
        self.strict = strict
        self.df = self._to_dataframe()

    @classmethod
    def from_file(cls, filepath: str, *, strict: bool = True) -> "MetadataJsonLoader":
        """
        Create a loader from a JSON file on disk.

        Args:
            filepath: Path to the JSON file.
            strict: Validation behavior; when True, enforces required fields and unique UUIDs.

        Returns:
            MetadataJsonLoader instance.
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data, strict=strict)

    @classmethod
    def from_str(cls, json_str: str, *, strict: bool = True) -> "MetadataJsonLoader":
        """
        Create a loader from a JSON string.

        Args:
            json_str: Raw JSON content as a string.
            strict: Validation behavior; when True, enforces required fields and unique UUIDs.

        Returns:
            MetadataJsonLoader instance.
        """
        return cls(json.loads(json_str), strict=strict)

    def _to_dataframe(self) -> pd.DataFrame:
        """
        Convert the provided JSON-like input to a normalized pandas DataFrame.

        Returns:
            DataFrame indexed by 'uuid' with flattened configuration columns.
        """
        data = self.json_data

        # Normalize multiple input shapes into list-of-records with uuid, label, config
        records = self._normalize_to_records(data)
        if not records:
            # Graceful empty
            return pd.DataFrame(columns=["label"]).rename_axis("uuid")

        # Build DataFrame
        base = pd.DataFrame.from_records(records)
        # Ensure required columns (uuid always required; label/config optional if not strict)
        if "uuid" not in base.columns:
            raise KeyError("Missing required field 'uuid'")
        if self.strict and "label" not in base.columns:
            raise KeyError("Missing required field 'label' in strict mode")
        if "label" not in base.columns:
            base["label"] = None
        if "config" not in base.columns:
            base["config"] = None

        # Flatten config into columns (prefix to avoid collisions)
        cfg = pd.json_normalize(base["config"].apply(lambda x: x or {}))
        if any(c in base.columns for c in cfg.columns):
            cfg = cfg.add_prefix("config.")
        df = pd.concat([base.drop(columns=["config"]), cfg], axis=1)

        # Validate UUIDs
        if df["uuid"].isna().any():
            if self.strict:
                raise ValueError("Some rows have missing UUIDs.")
            df = df.dropna(subset=["uuid"])  # drop rows with missing UUIDs

        # Enforce uniqueness
        if not df["uuid"].is_unique:
            if self.strict:
                dups = df["uuid"][df["uuid"].duplicated()].unique().tolist()
                raise ValueError(
                    f"UUIDs are not unique: {dups[:5]}{'...' if len(dups) > 5 else ''}"
                )
            # keep first occurrence
            df = df[~df["uuid"].duplicated(keep="first")]

        return df.set_index("uuid")

    def _normalize_to_records(self, data: Any) -> List[Dict[str, Any]]:
        """
        Normalize supported input shapes into a list of record dicts with
        keys: 'uuid', 'label', and 'config'.

        Args:
            data: JSON-like input in supported forms.

        Returns:
            A list of uniform record dictionaries.
        """
        # Case 1: list of records
        if isinstance(data, list):
            return [
                self._normalize_record(rec) for rec in data if isinstance(rec, dict)
            ]

        # Case 2: dict inputs
        if isinstance(data, dict):
            # 2a: dict of lists (columnar)
            if all(isinstance(v, list) for v in data.values()) and {
                "uuid",
                "label",
                "config",
            }.issubset(data.keys()):
                lengths = {k: len(v) for k, v in data.items() if isinstance(v, list)}
                if len(set(lengths.values())) > 1:
                    if self.strict:
                        raise ValueError(f"Column lengths differ: {lengths}")
                    # Fallback to shortest length
                n = min(lengths.values()) if lengths else 0
                out: List[Dict[str, Any]] = []
                for i in range(n):
                    out.append(
                        self._normalize_record(
                            {
                                "uuid": data.get("uuid", [None] * n)[i],
                                "label": data.get("label", [None] * n)[i],
                                "config": data.get("config", [None] * n)[i],
                            }
                        )
                    )
                return out

            # 2b: dict of dicts with string indices
            if all(isinstance(v, dict) for v in data.values()) and {
                "uuid",
                "label",
                "config",
            }.issubset(data.keys()):
                keys = (
                    set(data["uuid"])
                    .intersection(data["label"])
                    .intersection(data["config"])
                )
                # numeric-sort if possible
                ordered_keys = sorted(
                    keys, key=lambda k: int(k) if str(k).isdigit() else str(k)
                )
                return [
                    self._normalize_record(
                        {
                            "uuid": data["uuid"].get(k),
                            "label": data["label"].get(k),
                            "config": data["config"].get(k),
                        }
                    )
                    for k in ordered_keys
                ]

            # 2c: already record-like dict (single)
            if "uuid" in data or "label" in data or "config" in data:
                return [self._normalize_record(data)]

        # Unknown shape
        if self.strict:
            raise TypeError("Unsupported JSON structure for metadata")
        return []

    @staticmethod
    def _normalize_record(rec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure a single record has the shape {uuid, label, config},
        coercing 'config' to a dictionary where possible.

        Args:
            rec: Input record.

        Returns:
            Normalized record dict.
        """
        # Ensure config is a dict (or None)
        cfg = rec.get("config")
        if cfg is None:
            cfg = {}
        elif not isinstance(cfg, dict):
            # Attempt to parse if it's a JSON string
            if isinstance(cfg, str):
                try:
                    parsed = json.loads(cfg)
                    if isinstance(parsed, dict):
                        cfg = parsed
                    else:
                        cfg = {"value": parsed}
                except Exception:
                    cfg = {"value": cfg}
            else:
                cfg = {"value": cfg}
        return {
            "uuid": rec.get("uuid"),
            "label": rec.get("label"),
            "config": cfg,
        }

    # ------- Convenience -------
    def to_df(self, copy: bool = True) -> pd.DataFrame:
        """
        Return the underlying DataFrame.

        Args:
            copy: When True, returns a copy; otherwise returns a view/reference.

        Returns:
            pandas DataFrame indexed by 'uuid'.
        """
        return self.df.copy() if copy else self.df

    def head(self, n: int = 5) -> pd.DataFrame:
        """
        Convenience wrapper for DataFrame.head.

        Args:
            n: Number of rows to return.

        Returns:
            Top n rows of the metadata DataFrame.
        """
        return self.df.head(n)

    # ------- Lookups -------
    def get_by_uuid(self, uuid: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a row by UUID as a dictionary.

        Args:
            uuid: UUID key (index) to look up.

        Returns:
            Row as a dict, or None if not present.
        """
        if uuid not in self.df.index:
            return None
        return self.df.loc[uuid].to_dict()

    def get_by_label(self, label: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve the first row matching a label as a dictionary.

        Args:
            label: Label value to search for.

        Returns:
            Row as a dict, or None if not found.
        """
        row = self.df[self.df["label"] == label]
        return None if row.empty else row.iloc[0].to_dict()

    def join_with(self, other_df: pd.DataFrame, how: str = "inner") -> pd.DataFrame:
        """
        Join the metadata DataFrame with another DataFrame on the 'uuid' index.

        Args:
            other_df: DataFrame to join with (must be indexed compatibly).
            how: Join strategy (e.g., 'inner', 'left', 'outer').

        Returns:
            Joined pandas DataFrame.
        """
        return self.df.join(other_df, how=how)

    # ------- Filtering helpers -------
    def filter_by_uuid(self, uuids: Iterable[str]) -> pd.DataFrame:
        """
        Filter rows by a set or sequence of UUIDs.

        Args:
            uuids: Iterable of UUID strings to retain.

        Returns:
            Filtered DataFrame.
        """
        uuids_set = set(uuids)
        return self.df[self.df.index.isin(uuids_set)]

    def filter_by_label(self, labels: Iterable[str]) -> pd.DataFrame:
        """
        Filter rows by a set or sequence of labels.

        Args:
            labels: Iterable of label strings to retain.

        Returns:
            Filtered DataFrame.
        """
        labels_set = set(labels)
        return self.df[self.df["label"].isin(labels_set)]

    # ------- Introspection -------
    def list_uuids(self) -> List[str]:
        """
        Return a list of UUIDs present in the metadata index.

        Returns:
            List of UUID strings.
        """
        return list(self.df.index.astype(str))

    def list_labels(self) -> List[str]:
        """
        Return all non-null labels.

        Returns:
            List of label strings.
        """
        return list(self.df["label"].dropna().astype(str))
