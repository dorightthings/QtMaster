"""CPU-only protocol checks; no external data, provider or GPU required."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch
import yaml

from qtmaster.metrics import daily_metrics, summarize_seeds
from qtmaster.train import (
    TrainingConfig, evaluate, main, make_loader, official_type3_lr, parse_args, synthetic_splits,
)


class TrainingTests(unittest.TestCase):
    def test_current_pool_defaults_and_five_seeds(self):
        root = Path(__file__).resolve().parents[1]
        self.assertEqual(parse_args([]).seeds, [0, 1, 2, 3, 4])
        for pool, lr, cycle in (("csi300", 0.001, 7), ("csi800_direct", 0.0005, 3)):
            config = yaml.safe_load((root / "configs" / (pool + ".yaml")).read_text())
            self.assertEqual(config["dataset"], pool)
            self.assertEqual(config["model"]["cycle"], cycle)
            self.assertEqual(TrainingConfig(**config["training"]), TrainingConfig(learning_rate=lr))

    def test_original_type3_schedule(self):
        np.testing.assert_allclose(
            [official_type3_lr(0.001, epoch) for epoch in range(1, 7)],
            [0.001, 0.001, 0.001, 0.0008, 0.00064, 0.000512],
        )

    def test_daily_population_std_and_missing_labels(self):
        index = pd.MultiIndex.from_product(
            [pd.date_range("2020-01-01", periods=3), ["a", "b", "c", "d"]],
            names=["datetime", "instrument"],
        )
        values = np.array([1, 3, 2, 4, 3, 2, 4, 1, 4, 1, 2, 3], dtype=float)
        labels = np.array([1, 2, 3, 4] * 3, dtype=float)
        labels[-1] = np.nan
        result, daily = daily_metrics(pd.Series(values, index=index), labels)
        for name, ratio in (("IC", "ICIR"), ("RankIC", "RankICIR")):
            self.assertAlmostEqual(result[name], daily[name].mean())
            self.assertAlmostEqual(result[ratio], daily[name].mean() / daily[name].std(ddof=0))

    def test_missing_backtest_is_not_zero(self):
        summary = summarize_seeds([{"metrics": {"IC": 0.1, "AR": None}}, {"metrics": {"IC": 0.3, "AR": None}}])
        self.assertAlmostEqual(summary["IC"]["mean"], 0.2)
        self.assertAlmostEqual(summary["IC"]["std"], 0.1)
        self.assertIsNone(summary["AR"]["mean"])
        self.assertEqual(summary["AR"]["available_seeds"], 0)

    def test_complete_evaluation_masks_missing_labels(self):
        split = synthetic_splits()["valid"]
        split.dataset.labels[1] = float("nan")
        config = TrainingConfig(eval_batch_size=17, num_workers=0)

        class Zero(torch.nn.Module):
            def forward(self, x):
                return x.new_zeros((len(x), 1, 1))

        loader = make_loader(split, config)
        mse, prediction, audit = evaluate(Zero(), loader, len(split.dataset), torch.device("cpu"))
        expected = np.nanmean(split.dataset.labels.numpy().astype(np.float64) ** 2)
        self.assertAlmostEqual(mse, expected, places=6)
        self.assertEqual(audit["finite_labels"], len(split.dataset) - 1)
        self.assertEqual(len(prediction), len(split.dataset))

    def test_cli_smoke_records_artifacts_without_financial_metrics(self):
        root = Path(__file__).resolve().parents[1]
        output_parent = root / "outputs"
        output_parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="test-training-", dir=str(output_parent)) as output:
            code = main(["--config", str(root / "configs/csi300.yaml"), "--smoke-test",
                         "--seeds", "0", "--output-root", output])
            self.assertEqual(code, 0)
            experiment, = list(Path(output).iterdir())
            run = json.loads((experiment / "run.json").read_text())
            self.assertEqual(run["status"], "DONE")
            self.assertEqual(run["exit_code"], 0)
            self.assertEqual(run["config"]["model_id"], "market_conditioned_lla")
            self.assertEqual(run["config"]["parameters"], 213_279)
            self.assertEqual(run["config"]["optimizer_groups"], 1)
            self.assertEqual(run["argv"][1:3], ["-m", "qtmaster.train"])
            self.assertIn("--smoke-test", run["argv"])
            self.assertIn(str(root / "configs/csi300.yaml"), run["argv"])
            command = json.loads((experiment / "seed_0/command.json").read_text())
            self.assertEqual(command["argv"], run["argv"])
            self.assertEqual(command["shell_command"], run["command"])
            self.assertIn("-m qtmaster.train", run["command"])
            result = json.loads((experiment / "seed_0/result.json").read_text())
            self.assertEqual(result["kind"], "SYNTHETIC_SMOKE_ONLY")
            self.assertEqual(result["metrics"], {})
            self.assertFalse(result["selection"]["test_used_for_selection"])
            self.assertEqual(result["selection"]["epochs_completed"], 2)
            for name in ("checkpoint.pt", "environment.json", "config.json", "command.json",
                         "history.json", "valid_prediction.csv", "test_prediction.csv", "status.json", "train.log"):
                self.assertTrue((experiment / "seed_0" / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
