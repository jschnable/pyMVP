"""Lazy map columns and binary map serialization, independent of matrix storage."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd

def _pack_chromosome_groups(
    chrom_groups: Dict[str, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.array(list(chrom_groups.keys()), dtype=str)
    offsets = np.zeros(len(order) + 1, dtype=np.int64)
    for idx, chrom in enumerate(order):
        offsets[idx + 1] = offsets[idx] + int(len(chrom_groups[str(chrom)]))
    flat_indices = np.empty(int(offsets[-1]), dtype=np.int64)
    for idx, chrom in enumerate(order):
        start, end = int(offsets[idx]), int(offsets[idx + 1])
        flat_indices[start:end] = np.asarray(chrom_groups[str(chrom)], dtype=np.int64)
    return order, offsets, flat_indices


def _unpack_chromosome_groups(
    order: Sequence[str],
    offsets: np.ndarray,
    flat_indices: np.ndarray,
) -> Dict[str, np.ndarray]:
    chrom_groups: Dict[str, np.ndarray] = {}
    for idx, chrom in enumerate(order):
        start = int(offsets[idx])
        end = int(offsets[idx + 1])
        chrom_groups[str(chrom)] = np.asarray(flat_indices[start:end], dtype=np.int64)
    return chrom_groups


class _PackedUtf8Column:
    """Lazy UTF-8 string column stored as a byte blob plus offsets."""

    def __init__(
        self,
        offsets: np.ndarray,
        data: np.ndarray,
        indexer: Optional[np.ndarray] = None,
    ):
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self.data = np.asarray(data, dtype=np.uint8)
        self._indexer = None if indexer is None else np.asarray(indexer, dtype=np.int64)
        self._decoded: Optional[np.ndarray] = None

    def __len__(self) -> int:
        if self._indexer is not None:
            return int(self._indexer.size)
        return max(0, int(self.offsets.size) - 1)

    def take(self, indices: np.ndarray) -> "_PackedUtf8Column":
        idx = np.asarray(indices, dtype=np.int64)
        if self._indexer is not None:
            idx = self._indexer[idx]
        return _PackedUtf8Column(self.offsets, self.data, indexer=idx)

    def to_numpy(self) -> np.ndarray:
        if self._decoded is None:
            buffer = memoryview(self.data)
            n = len(self)
            out = np.empty(n, dtype=object)
            parent_idx = (
                np.arange(n, dtype=np.int64) if self._indexer is None else self._indexer
            )
            for i in range(n):
                src = int(parent_idx[i])
                start = int(self.offsets[src])
                end = int(self.offsets[src + 1])
                out[i] = bytes(buffer[start:end]).decode("utf-8")
            self._decoded = out
        return self._decoded


class _CategoricalUtf8Column:
    """Lazy low-cardinality string column stored as category codes."""

    def __init__(self, codes: np.ndarray, categories: np.ndarray):
        self.codes = np.asarray(codes, dtype=np.int32)
        self.categories = np.asarray(categories, dtype=object)
        self._decoded: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return int(self.codes.size)

    def take(self, indices: np.ndarray) -> "_CategoricalUtf8Column":
        idx = np.asarray(indices, dtype=np.int64)
        return _CategoricalUtf8Column(self.codes[idx], self.categories)

    def to_numpy(self) -> np.ndarray:
        if self._decoded is None:
            self._decoded = np.asarray(self.categories[self.codes], dtype=object)
        return self._decoded


def _pack_utf8_column(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Encode a string column as UTF-8 bytes with offsets."""
    encoded_values = [str(value).encode("utf-8") for value in values]
    lengths = np.fromiter(
        (len(value) for value in encoded_values),
        dtype=np.int64,
        count=len(encoded_values),
    )
    offsets = np.empty(len(encoded_values) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(lengths, out=offsets[1:])
    data = np.frombuffer(b"".join(encoded_values), dtype=np.uint8)
    return offsets, data


def _materialize_lazy_column(
    column: Union[np.ndarray, "_PackedUtf8Column", "_CategoricalUtf8Column"]
) -> np.ndarray:
    if hasattr(column, "to_numpy"):
        return column.to_numpy()  # type: ignore[return-value]
    return np.asarray(column)


def save_genotype_map_cache(
    cache_path: Union[str, Path],
    map_data: Union[pd.DataFrame, "GenotypeMap"],
) -> None:
    """Persist genotype-map metadata to a fast binary cache."""
    from .data_types import (GenotypeMap, canonicalize_genotype_map_dataframe,
        attach_genotype_map_metadata, group_marker_indices_by_labels,
        LEGACY_MARKER_ID_COLUMN, MARKER_ID_COLUMN, CHROM_COLUMN)
    if isinstance(map_data, GenotypeMap):
        map_df = map_data.to_dataframe()
        metadata = map_data.metadata
    else:
        map_df = map_data.copy()
        metadata = getattr(map_df, "attrs", {})

    map_df = canonicalize_genotype_map_dataframe(map_df)
    map_df = attach_genotype_map_metadata(map_df)

    arrays: Dict[str, np.ndarray] = {
        "format_version": np.asarray(2, dtype=np.int16),
        "columns": np.asarray(map_df.columns, dtype=str),
        "is_imputed": np.asarray(bool(metadata.get("is_imputed", map_df.attrs.get("is_imputed", False)))),
    }
    column_kinds = np.empty(len(map_df.columns), dtype="<U24")
    alias_of = np.full(len(map_df.columns), -1, dtype=np.int64)

    for idx, column in enumerate(map_df.columns):
        series = map_df[column]
        values = series.to_numpy(copy=False)
        if (
            column == LEGACY_MARKER_ID_COLUMN
            and MARKER_ID_COLUMN in map_df.columns
            and series.astype(str).equals(map_df[MARKER_ID_COLUMN].astype(str))
        ):
            column_kinds[idx] = "alias"
            alias_of[idx] = int(list(map_df.columns).index(MARKER_ID_COLUMN))
            continue

        if pd.api.types.is_bool_dtype(series.dtype) or pd.api.types.is_numeric_dtype(series.dtype):
            column_kinds[idx] = "numeric"
            arrays[f"col_{idx}"] = np.asarray(values)
            continue

        string_values = series.astype(str).to_numpy(dtype=object, copy=False)
        if column == CHROM_COLUMN:
            categories, codes = np.unique(string_values, return_inverse=True)
            column_kinds[idx] = "categorical_utf8"
            arrays[f"col_{idx}_codes"] = np.asarray(codes, dtype=np.int32)
            arrays[f"col_{idx}_categories"] = np.asarray(categories, dtype=str)
            continue

        column_kinds[idx] = "packed_utf8"
        offsets, data = _pack_utf8_column(string_values)
        arrays[f"col_{idx}_offsets"] = offsets
        arrays[f"col_{idx}_data"] = data

    arrays["column_kinds"] = column_kinds
    arrays["column_alias_of"] = alias_of

    chrom_groups = map_df.attrs.get("chromosome_groups") or metadata.get("chromosome_groups")
    if chrom_groups is None:
        chrom_groups = group_marker_indices_by_labels(
            np.asarray(map_df[CHROM_COLUMN]).astype(str, copy=False)
        )
    order, offsets, flat_indices = _pack_chromosome_groups(chrom_groups)
    arrays["chrom_group_keys"] = order
    arrays["chrom_group_offsets"] = offsets
    arrays["chrom_group_indices"] = flat_indices

    np.savez(cache_path, **arrays)


def load_genotype_map_cache(
    cache_path: Union[str, Path],
    *,
    legacy_csv_path: Optional[Union[str, Path]] = None,
    migrate_legacy: bool = False,
    legacy_is_imputed: Optional[bool] = None,
) -> "GenotypeMap":
    """Load genotype-map metadata from a binary cache or legacy CSV cache."""
    from .data_types import GenotypeMap, canonicalize_genotype_map_dataframe, attach_genotype_map_metadata
    cache_path = Path(cache_path)
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as archive:
            if "format_version" in archive and "column_kinds" in archive:
                columns = archive["columns"].astype(str).tolist()
                column_kinds = archive["column_kinds"].astype(str).tolist()
                if "column_alias_of" in archive:
                    alias_of = np.asarray(archive["column_alias_of"], dtype=np.int64)
                else:
                    alias_of = np.full(len(columns), -1, dtype=np.int64)
                column_data: Dict[str, Any] = {}
                for idx, column in enumerate(columns):
                    kind = column_kinds[idx]
                    if kind == "alias":
                        source_idx = int(alias_of[idx])
                        if source_idx < 0:
                            raise ValueError(f"Invalid alias mapping for cached column '{column}'")
                        column_data[column] = column_data[columns[source_idx]]
                    elif kind == "numeric":
                        column_data[column] = archive[f"col_{idx}"]
                    elif kind == "packed_utf8":
                        column_data[column] = _PackedUtf8Column(
                            archive[f"col_{idx}_offsets"],
                            archive[f"col_{idx}_data"],
                        )
                    elif kind == "categorical_utf8":
                        column_data[column] = _CategoricalUtf8Column(
                            archive[f"col_{idx}_codes"],
                            archive[f"col_{idx}_categories"],
                        )
                    else:
                        raise ValueError(f"Unknown cached column kind: {kind}")

                metadata: Dict[str, Any] = {}
                if "is_imputed" in archive:
                    metadata["is_imputed"] = bool(np.asarray(archive["is_imputed"]).item())
                if (
                    "chrom_group_keys" in archive
                    and "chrom_group_offsets" in archive
                    and "chrom_group_indices" in archive
                ):
                    order = archive["chrom_group_keys"].astype(str).tolist()
                    offsets = np.asarray(archive["chrom_group_offsets"], dtype=np.int64)
                    flat = np.asarray(archive["chrom_group_indices"], dtype=np.int64)
                    metadata["chromosome_order"] = order
                    metadata["chromosome_groups"] = _unpack_chromosome_groups(
                        order,
                        offsets,
                        flat,
                    )
                return GenotypeMap.from_columns(
                    column_data,
                    column_order=columns,
                    metadata=metadata,
                )

            if legacy_csv_path is None:
                columns = archive["columns"].astype(str).tolist()
                frame_data = {
                    column: archive[f"col_{idx}"]
                    for idx, column in enumerate(columns)
                }
                map_df = canonicalize_genotype_map_dataframe(pd.DataFrame(frame_data, copy=False))
                attach_genotype_map_metadata(map_df)
                if "is_imputed" in archive:
                    map_df.attrs["is_imputed"] = bool(np.asarray(archive["is_imputed"]).item())
                return GenotypeMap(map_df, metadata=dict(map_df.attrs))

    if legacy_csv_path is None:
        raise FileNotFoundError(f"Genotype map cache not found: {cache_path}")

    legacy_path = Path(legacy_csv_path)
    if not legacy_path.exists():
        raise FileNotFoundError(f"Genotype map cache not found: {cache_path}")

    map_df = canonicalize_genotype_map_dataframe(pd.read_csv(legacy_path))
    map_df = attach_genotype_map_metadata(map_df)
    if legacy_is_imputed is not None:
        map_df.attrs["is_imputed"] = bool(legacy_is_imputed)
    if migrate_legacy:
        try:
            save_genotype_map_cache(cache_path, map_df)
        except Exception:
            pass
    return GenotypeMap(map_df, metadata=dict(map_df.attrs))
