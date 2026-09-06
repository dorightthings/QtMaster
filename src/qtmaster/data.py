"""Portable loader for the frozen eight-day stock/market sampler bundles.

Pickle is executable input: both explicit caller consent and manifest verification
are required. Checksums identify bytes; they do not make an untrusted pickle safe.
"""

from __future__ import annotations

import hashlib
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

LOOKBACK = 8
STOCK_FEATURES = 158
MARKET_FEATURES = 63
MODEL_INPUT_FEATURES = 221
RAW_WIDTH = 222
STAGES = ("train", "valid", "test")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: str | Path) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "qtmaster-data-v1":
        raise ValueError("Unsupported data manifest schema")
    if manifest.get("sample_shape") != [LOOKBACK, RAW_WIDTH]:
        raise ValueError("Data manifest must describe [8,222] raw windows")
    return manifest


def verify_dataset_files(
    data_root: str | Path, dataset: str, manifest: Mapping[str, Any]
) -> dict[str, Path]:
    """Verify every split before any pickle in this dataset is opened."""
    root = Path(data_root).resolve()
    if dataset not in manifest["datasets"]:
        raise ValueError(f"Unknown dataset: {dataset}")
    paths: dict[str, Path] = {}
    for stage in STAGES:
        spec = manifest["datasets"][dataset]["files"][stage]
        relative = Path(spec["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Manifest data paths must stay inside data_root")
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            raise ValueError("Data file resolves outside data_root") from None
        if not path.is_file():
            raise FileNotFoundError(f"Missing {dataset}/{stage}: {path}; see data/README.md")
        if path.stat().st_size != int(spec["bytes"]):
            raise ValueError(f"Byte-size mismatch for {dataset}/{stage}")
        if sha256_file(path) != spec["sha256"]:
            raise ValueError(f"SHA256 mismatch for {dataset}/{stage}")
        paths[stage] = path
    return paths


def current_labels(sampler: Any) -> np.ndarray:
    """Read the label at each sample's last, observed date (column 221)."""
    idx_map = np.asarray(sampler.idx_map)
    rows = idx_map[:, 0].astype(np.int64, copy=False)
    columns = idx_map[:, 1].astype(np.int64, copy=False)
    current = np.asarray(sampler.idx_arr[rows, columns])
    if not np.isfinite(current).all():
        raise ValueError("Sampler current positions must be finite")
    positions = current.astype(np.int64)
    if not np.array_equal(current, positions):
        raise ValueError("Sampler current positions must be exact integers")
    return np.asarray(sampler.data_arr[positions, MODEL_INPUT_FEATURES], dtype=np.float32)


def daily_ranges(index: pd.MultiIndex) -> list[tuple[int, int]]:
    dates = pd.DatetimeIndex(index.get_level_values("datetime"))
    if not len(dates):
        return []
    changes = np.flatnonzero(dates.values[1:] != dates.values[:-1]) + 1
    edges = np.concatenate(([0], changes, [len(dates)]))
    ranges = [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])]
    if len(ranges) != dates.nunique():
        raise ValueError("Datetime index must use contiguous daily blocks")
    return ranges


def make_train_view(
    sampler: Any, source_positions: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, pd.MultiIndex, dict[str, Any]]:
    """Original float32 daily finite/tail-drop/retained-ddof=1 protocol."""
    if source_positions is None:
        source_positions = np.arange(len(sampler.get_index()), dtype=np.int64)
    source_positions = np.asarray(source_positions, dtype=np.int64)
    source_index = sampler.get_index()[source_positions]
    source_labels = current_labels(sampler)[source_positions]
    retained_positions: list[np.ndarray] = []
    retained_labels: list[np.ndarray] = []
    removed_extremes = 0
    for start, end in daily_ranges(source_index):
        raw = torch.from_numpy(source_labels[start:end].copy())
        finite_local = torch.nonzero(torch.isfinite(raw), as_tuple=False).reshape(-1)
        finite_values = raw[finite_local]
        if len(finite_values) < 2:
            raise ValueError("Training day has fewer than two finite labels")
        k = int(math.floor(len(finite_values) * 0.025))
        order = torch.argsort(finite_values)
        ranked_keep = order if k == 0 else order[k:-k]
        kept_local = torch.sort(finite_local[ranked_keep]).values
        kept_values = raw[kept_local]
        std = kept_values.std(unbiased=True)
        if not torch.isfinite(std) or float(std) == 0.0:
            raise ValueError("Training day has zero/non-finite retained label std")
        normalized = (kept_values - kept_values.mean()) / std
        if not torch.isfinite(normalized).all():
            raise ValueError("Training label normalization produced non-finite values")
        retained_positions.append(source_positions[start + kept_local.numpy()])
        retained_labels.append(normalized.numpy().astype(np.float32, copy=False))
        removed_extremes += 2 * k
    if not retained_positions:
        raise ValueError("Empty training dataset")
    positions = np.concatenate(retained_positions).astype(np.int64, copy=False)
    labels = np.concatenate(retained_labels).astype(np.float32, copy=False)
    metadata = {
        "rule": "daily finite labels; drop each tail floor(n_finite*0.025); retained zscore ddof=1",
        "source_samples": len(source_positions),
        "retained_samples": len(positions),
        "removed_nonfinite": int((~np.isfinite(source_labels)).sum()),
        "removed_extremes": removed_extremes,
    }
    return positions, labels, sampler.get_index()[positions], metadata


def normalize_labels_by_day(raw_labels: np.ndarray, index: pd.MultiIndex) -> np.ndarray:
    """Daily ddof=1 z-score without tail removal; missing labels remain NaN."""
    raw_labels = np.asarray(raw_labels, dtype=np.float32)
    normalized = np.full(len(raw_labels), np.nan, dtype=np.float32)
    for start, end in daily_ranges(index):
        values = torch.from_numpy(raw_labels[start:end].copy())
        mask = torch.isfinite(values)
        kept = values[mask]
        if len(kept) < 2:
            raise ValueError("Evaluation day has fewer than two finite labels")
        std = kept.std(unbiased=True)
        if not torch.isfinite(std) or float(std) == 0.0:
            raise ValueError("Evaluation day has zero/non-finite label std")
        normalized[start:end][mask.numpy()] = ((kept - kept.mean()) / std).numpy()
    return normalized


def build_exact_window_indices(sampler: Any, source_positions: np.ndarray) -> np.ndarray:
    """Cache the original TSDataSampler padding/fill indices without changing data."""
    source = np.asarray(source_positions, dtype=np.int64)
    step_len = int(sampler.step_len)
    if step_len != LOOKBACK:
        raise ValueError("Expected an eight-day sampler")
    idx_map = np.asarray(sampler.idx_map)
    idx_arr = np.asarray(sampler.idx_arr, dtype=np.float64)
    if np.any(source < 0) or np.any(source >= len(idx_map)):
        raise IndexError("Source position outside sampler")
    if int(sampler.nan_idx) > np.iinfo(np.int32).max:
        raise ValueError("Sampler nan_idx does not fit int32")
    cached = np.empty((len(source), step_len), dtype=np.int32)
    offsets = np.arange(1 - step_len, 1, dtype=np.int64)
    column_offsets = np.arange(step_len, dtype=np.int64)[None, :]
    fillna_type = str(sampler.fillna_type)
    if fillna_type not in ("none", "ffill", "ffill+bfill"):
        raise ValueError(f"Unsupported sampler fill: {fillna_type}")
    for start in range(0, len(source), 262_144):
        end = min(start + 262_144, len(source))
        row_col = idx_map[source[start:end]]
        rows = row_col[:, 0].astype(np.int64, copy=False)[:, None] + offsets[None, :]
        cols = row_col[:, 1].astype(np.int64, copy=False)[:, None]
        values = idx_arr[np.maximum(rows, 0), cols]
        values[rows < 0] = np.nan
        if fillna_type in ("ffill", "ffill+bfill"):
            take = np.where(~np.isnan(values), column_offsets, 0)
            np.maximum.accumulate(take, axis=1, out=take)
            values = np.take_along_axis(values, take, axis=1)
        if fillna_type == "ffill+bfill":
            reversed_values = values[:, ::-1]
            take = np.where(~np.isnan(reversed_values), column_offsets, 0)
            np.maximum.accumulate(take, axis=1, out=take)
            values = np.take_along_axis(reversed_values, take, axis=1)[:, ::-1]
        integer = np.nan_to_num(values, nan=int(sampler.nan_idx)).astype(np.int64)
        if integer.size and (integer.min() < 0 or integer.max() > np.iinfo(np.int32).max):
            raise ValueError("Invalid cached window index")
        cached[start:end] = integer.astype(np.int32, copy=False)
    cached.setflags(write=False)
    return cached


class StockSequenceDataset(Dataset):
    def __init__(self, sampler: Any, positions: np.ndarray, labels: np.ndarray):
        self.sampler = sampler
        self.positions = np.asarray(positions, dtype=np.int64)
        self.labels = np.asarray(labels, dtype=np.float32)
        if len(self.positions) != len(self.labels):
            raise ValueError("Position/label length mismatch")
        self.window_indices = build_exact_window_indices(sampler, self.positions)

    def __len__(self) -> int:
        return len(self.positions)

    def _features(self, local: np.ndarray) -> np.ndarray:
        raw = np.asarray(self.sampler.data_arr[self.window_indices[local]])
        if raw.shape != (len(local), LOOKBACK, RAW_WIDTH) or raw.dtype != np.float32:
            raise ValueError(f"Unexpected raw sample shape/dtype: {raw.shape}/{raw.dtype}")
        # Exclude the entire historical label column, not just its last timestep.
        features = np.ascontiguousarray(raw[:, :, :MODEL_INPUT_FEATURES], dtype=np.float32)
        if not np.isfinite(features).all():
            raise ValueError("Stock/market features contain NaN or infinity")
        return features

    def __getitem__(self, local_position: int) -> tuple[np.ndarray, np.float32, int]:
        features = self._features(np.asarray([local_position], dtype=np.int64))[0]
        return features, np.float32(self.labels[local_position]), int(local_position)

    def __getitems__(self, local_positions: Sequence[int]) -> tuple[torch.Tensor, ...]:
        local = np.asarray(local_positions, dtype=np.int64)
        if local.ndim != 1 or not len(local):
            raise ValueError("Expected a nonempty one-dimensional batch")
        return (
            torch.from_numpy(self._features(local)),
            torch.from_numpy(np.ascontiguousarray(self.labels[local])),
            torch.from_numpy(np.ascontiguousarray(local)),
        )


def identity_batch_collate(batch: Any) -> Any:
    """DataLoader adapter for the already-collated __getitems__ result."""
    return batch


@dataclass
class PreparedSplit:
    dataset: Dataset
    index: pd.MultiIndex
    labels: np.ndarray
    raw_labels: np.ndarray
    metadata: dict[str, Any]
    sampler_index: pd.MultiIndex | None = None
    kept_positions: np.ndarray | None = None


def prepare_split(sampler: Any, stage: str, spec: Mapping[str, Any] | None = None) -> PreparedSplit:
    if stage not in STAGES:
        raise ValueError(f"Unknown split: {stage}")
    index = sampler.get_index()
    if not isinstance(index, pd.MultiIndex) or list(index.names) != ["datetime", "instrument"]:
        raise ValueError("Expected MultiIndex ['datetime', 'instrument']")
    if index.has_duplicates or not index.is_monotonic_increasing or not len(index):
        raise ValueError("Sampler index must be nonempty, unique, and sorted")
    dates = pd.DatetimeIndex(index.get_level_values("datetime"))
    if spec is not None:
        actual = {
            "samples": len(index), "dates": len(daily_ranges(index)),
            "date_min": str(dates.min().date()), "date_max": str(dates.max().date()),
        }
        if actual != {key: spec[key] for key in actual}:
            raise ValueError(f"Sampler metadata mismatch: {stage}")
    all_raw = current_labels(sampler)
    if stage == "train":
        positions, labels, retained_index, metadata = make_train_view(sampler)
    else:
        positions = np.arange(len(index), dtype=np.int64)
        retained_index = index
        labels = normalize_labels_by_day(all_raw, index)
        metadata = {"rule": "daily finite-label zscore ddof=1; no tail drop; missing labels retained"}
    dataset = StockSequenceDataset(sampler, positions, labels)
    # Small exactness check includes a padded early window and a late window.
    probe_local = np.unique([0, len(dataset) - 1]).astype(np.int64)
    original = np.asarray(sampler[positions[probe_local]])[:, :, :MODEL_INPUT_FEATURES]
    if not np.array_equal(dataset._features(probe_local), original):
        raise ValueError("Cached sampler differs from original feature windows")
    metadata.update({"stage": stage, "samples": len(dataset), "missing_labels": int(np.isnan(labels).sum())})
    return PreparedSplit(dataset, retained_index, labels, all_raw[positions], metadata, index, positions)


def load_splits(
    data_root: str | Path,
    dataset: str,
    trust_pickle: bool = False,
    manifest_path: str | Path | None = None,
) -> dict[str, PreparedSplit]:
    if not trust_pickle:
        raise ValueError("Pickle can execute code. Only use trusted bundles and explicitly set trust_pickle=True / --trust-pickle")
    root = Path(data_root)
    manifest = read_manifest(manifest_path or root / "manifest.json")
    paths = verify_dataset_files(root, dataset, manifest)
    result: dict[str, PreparedSplit] = {}
    for stage in STAGES:
        with paths[stage].open("rb") as handle:
            sampler = pickle.load(handle)
        result[stage] = prepare_split(sampler, stage, manifest["datasets"][dataset]["files"][stage])
    return result
