import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from qtmaster.data import (
    StockSequenceDataset,
    current_labels,
    daily_ranges,
    identity_batch_collate,
    load_splits,
    make_train_view,
    normalize_labels_by_day,
    prepare_split,
    verify_dataset_files,
)


class FakeSampler:
    """Minimal float32 sampler with the same observable index/array contract."""

    def __init__(self, labels, days=None):
        labels = np.asarray(labels, dtype=np.float32)
        n = len(labels)
        self.step_len = 8
        self.fillna_type = "ffill+bfill"
        self.nan_idx = n
        self.idx_map = np.column_stack((np.arange(n), np.zeros(n, dtype=int)))
        self.idx_arr = np.arange(n, dtype=float)[:, None]
        self.data_arr = np.zeros((n + 1, 222), dtype=np.float32)
        self.data_arr[:n, :221] = np.arange(n * 221, dtype=np.float32).reshape(n, 221)
        self.data_arr[:n, 221] = labels
        self.data_arr[n] = np.nan
        days = pd.to_datetime(days if days is not None else ["2020-01-02"] * n)
        self.index = pd.MultiIndex.from_arrays(
            (days, [f"S{i:04d}" for i in range(n)]), names=["datetime", "instrument"]
        )

    def get_index(self):
        return self.index

    def __getitem__(self, positions):
        local = np.asarray(positions)
        scalar = local.ndim == 0
        local = local.reshape(-1)
        windows = np.maximum(local[:, None] + np.arange(-7, 1)[None, :], 0)
        result = self.data_arr[windows]
        return result[0] if scalar else result


class DataTests(unittest.TestCase):
    def test_daily_train_tail_order_and_ddof(self):
        labels = np.arange(80, dtype=np.float32)[::-1].copy()
        sampler = FakeSampler(labels)
        positions, normalized, index, metadata = make_train_view(sampler)
        np.testing.assert_array_equal(positions, np.arange(2, 78))
        expected = torch.tensor(labels[2:78].copy())
        expected = (expected - expected.mean()) / expected.std(unbiased=True)
        np.testing.assert_array_equal(normalized, expected.numpy())
        self.assertEqual(metadata["removed_extremes"], 4)
        self.assertTrue(index.equals(sampler.index[positions]))

    def test_tail_count_uses_finite_count(self):
        values = np.arange(40, dtype=np.float32)
        values[-1] = np.nan
        positions, labels, _, meta = make_train_view(FakeSampler(values))
        self.assertEqual(len(positions), 39)  # floor(39*.025)=0
        self.assertEqual(meta["removed_extremes"], 0)
        self.assertTrue(np.isfinite(labels).all())

    def test_normalize_days_independently_keep_missing(self):
        values = np.array([1, 3, np.nan, 100, 104, np.inf], dtype=np.float32)
        sampler = FakeSampler(values, ["2020-01-02"] * 3 + ["2020-01-03"] * 3)
        normalized = normalize_labels_by_day(values, sampler.index)
        expected = np.array([-2**-0.5, 2**-0.5, np.nan] * 2, dtype=np.float32)
        np.testing.assert_allclose(normalized, expected, rtol=1e-6, equal_nan=True)
        self.assertEqual(daily_ranges(sampler.index), [(0, 3), (3, 6)])

    def test_constant_labels_rejected(self):
        sampler = FakeSampler([1, 1, 1])
        with self.assertRaisesRegex(ValueError, "std"):
            make_train_view(sampler)
        with self.assertRaisesRegex(ValueError, "std"):
            normalize_labels_by_day(current_labels(sampler), sampler.index)

    def test_feature_axis_and_label_isolation(self):
        sampler = FakeSampler(np.arange(12))
        dataset = StockSequenceDataset(sampler, np.arange(12), np.arange(12, dtype=np.float32))
        x_before, label, idx = dataset[9]
        self.assertEqual(x_before.shape, (8, 221))
        self.assertEqual(x_before.dtype, np.float32)
        np.testing.assert_array_equal(x_before, sampler[9][:, :221])
        sampler.data_arr[:, 221] = np.nan
        x_after, _, _ = dataset[9]
        np.testing.assert_array_equal(x_before, x_after)
        self.assertEqual((label, idx), (9, 9))

    def test_market_finiteness_checked(self):
        sampler = FakeSampler(np.arange(12))
        dataset = StockSequenceDataset(sampler, np.arange(12), np.arange(12))
        sampler.data_arr[9, 200] = np.inf
        with self.assertRaisesRegex(ValueError, "features"):
            dataset[9]

    def test_cached_batch_and_scalar_match(self):
        sampler = FakeSampler(np.arange(12))
        dataset = StockSequenceDataset(sampler, np.arange(12), np.arange(12))
        loader = DataLoader(dataset, batch_size=4, collate_fn=identity_batch_collate)
        x, y, local = next(iter(loader))
        np.testing.assert_array_equal(x.numpy(), np.stack([dataset[i][0] for i in range(4)]))
        np.testing.assert_array_equal(y.numpy(), np.arange(4))
        np.testing.assert_array_equal(local.numpy(), np.arange(4))

    def test_prepare_evaluation_retains_missing_rows(self):
        split = prepare_split(FakeSampler([0, 1, np.nan, 2, 3]), "valid")
        self.assertEqual(len(split.dataset), 5)
        self.assertTrue(np.isnan(split.labels[2]))
        self.assertEqual(split.metadata["missing_labels"], 1)

    def test_pickle_requires_explicit_consent(self):
        with patch("qtmaster.data.pickle.load") as unpickle:
            with self.assertRaisesRegex(ValueError, "Pickle can execute code"):
                load_splits("not-a-directory", "csi300")
            unpickle.assert_not_called()

    def test_hash_failure_precedes_any_unpickle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = {}
            for stage in ("train", "valid", "test"):
                path = root / f"{stage}.pkl"
                path.write_bytes(b"not a pickle")
                files[stage] = {"path": path.name, "bytes": 12, "sha256": "0" * 64}
            manifest = {"schema_version": "qtmaster-data-v1", "sample_shape": [8, 222], "datasets": {"csi300": {"files": files}}}
            (root / "manifest.json").write_text(json.dumps(manifest))
            with patch("qtmaster.data.pickle.load") as unpickle:
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    load_splits(root, "csi300", trust_pickle=True)
                unpickle.assert_not_called()

    def test_manifest_paths_cannot_escape_data_root(self):
        manifest = {"datasets": {"csi300": {"files": {"train": {"path": "../private.pkl"}}}}}
        with self.assertRaisesRegex(ValueError, "inside data_root"):
            verify_dataset_files(".", "csi300", manifest)


class ExportTests(unittest.TestCase):
    def test_archive_is_verified_and_never_overwrites(self):
        import tarfile

        script_path = Path(__file__).resolve().parents[1] / "scripts" / "export_data_bundle.py"
        spec = importlib.util.spec_from_file_location("export_data_bundle", script_path)
        export = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(export)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = {}
            for stage in ("train", "valid", "test"):
                path = root / "csi300" / f"{stage}.pkl"
                path.parent.mkdir(exist_ok=True)
                payload = f"opaque bytes {stage}".encode()
                path.write_bytes(payload)
                files[stage] = {"path": f"csi300/{stage}.pkl", "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
            manifest = {"schema_version": "qtmaster-data-v1", "datasets": {"csi300": {"files": files}}}
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            output = root / "bundle.tar"
            command = ["--source-root", str(root), "--manifest", str(manifest_path), "--output", str(output)]
            self.assertEqual(export.main(command), 0)
            previous = output.read_bytes()
            with tarfile.open(output) as archive:
                self.assertEqual(len(archive.getnames()), 5)
                actual = archive.extractfile("data/csi300/train.pkl").read()
                self.assertEqual(actual, b"opaque bytes train")
                bundle = json.loads(archive.extractfile("data/bundle_manifest.json").read())
                self.assertEqual(bundle["data_manifest_sha256"], hashlib.sha256(archive.extractfile("data/manifest.json").read()).hexdigest())
            with self.assertRaises(SystemExit):
                export.main(command)
            self.assertEqual(output.read_bytes(), previous)


if __name__ == "__main__":
    unittest.main()
