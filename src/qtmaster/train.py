"""Portable training entry point for MarketGate + LLA + PureMLP.

Runs seeds sequentially on the requested device. For separate GPUs, launch
separate commands with disjoint seeds; each command creates a fresh run folder.
"""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import getpass
import importlib.metadata
import json
import logging
import math
import os
from pathlib import Path
import platform
import random
import shlex
import socket
import subprocess
import sys
import time
import uuid

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import yaml

from .metrics import daily_metrics, summarize_seeds
from .model import ModelConfig, build_model


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 0.001
    batch_size: int = 64
    eval_batch_size: int = 64
    max_epochs: int = 30
    patience: int = 5
    drop_last: bool = True
    lr_schedule: str = "type3"
    num_workers: int = 2


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, payload):
    """Atomic progress updates inside this run's newly allocated directory."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(path))


def official_type3_lr(initial_lr, completed_epoch_one_based):
    """Schedule is called AFTER each epoch, as in the original training loop."""
    if completed_epoch_one_based < 3:
        return initial_lr
    return initial_lr * (0.8 ** (completed_epoch_one_based - 3))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def collate_batch(batch):
    # Both real and synthetic datasets expose an efficient batched __getitems__.
    return batch


def make_loader(split, config, training=False, device=None):
    return DataLoader(
        split.dataset,
        batch_size=config.batch_size if training else config.eval_batch_size,
        shuffle=training, drop_last=config.drop_last if training else False,
        num_workers=config.num_workers, collate_fn=collate_batch,
        pin_memory=device is not None and device.type == "cuda",
    )


def evaluate(model, loader, size, device, return_predictions=True):
    """All rows exactly once; masked finite-label SSE divided by finite count."""
    model.eval()
    seen = np.zeros(size, dtype=bool)
    predictions = np.empty(size, dtype=np.float32) if return_predictions else None
    sse, count = 0.0, 0
    with torch.no_grad():
        for features, labels, positions in loader:
            positions = positions.numpy().astype(np.int64, copy=False)
            if len(np.unique(positions)) != len(positions) or seen[positions].any():
                raise ValueError("Evaluation contains duplicate samples")
            seen[positions] = True
            output = model(features.to(device=device, dtype=torch.float32, non_blocking=True)).reshape(-1)
            if len(output) != len(positions) or not torch.isfinite(output).all():
                raise ValueError("Invalid model predictions")
            target = labels.to(device=device, dtype=torch.float32, non_blocking=True)
            finite = torch.isfinite(target)
            if finite.any():
                # Preserve float32 per-batch SSE, then sum scalars in Python float.
                diff = output[finite] - target[finite]
                sse += float(torch.sum(diff * diff).item())
                count += int(finite.sum().item())
            if predictions is not None:
                predictions[positions] = output.cpu().numpy()
    if not seen.all() or not count:
        raise ValueError("Evaluation must visit every row and contain finite labels")
    return sse / count, predictions, {"rows": int(seen.sum()), "finite_labels": count, "aggregation": "SSE/count"}


def train_model(model, train_loader, valid_loader, valid_size, config, device, run_dir, logger):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    checkpoint = run_dir / "checkpoint.pt"
    best_mse, best_epoch, bad_epochs = math.inf, 0, 0
    history = []
    for epoch in range(1, config.max_epochs + 1):
        started = time.monotonic()
        model.train()
        train_sse, count, steps = 0.0, 0, 0
        lr_used = float(optimizer.param_groups[0]["lr"])
        for features, labels, _ in train_loader:
            features = features.to(device=device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(device=device, dtype=torch.float32, non_blocking=True)
            if not torch.isfinite(labels).all():
                raise ValueError("Training labels must be finite after training preprocessing")
            prediction = model(features).reshape(-1)
            if prediction.shape != labels.shape:
                raise ValueError("Prediction/target shape mismatch")
            loss = torch.mean((prediction - labels) ** 2)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if steps == 0 and epoch == 1:
                gradients = [p.grad for p in model.parameters() if p.grad is not None]
                if not gradients or not all(torch.isfinite(g).all() for g in gradients):
                    raise ValueError("Missing or non-finite training gradients")
            optimizer.step()
            train_sse += float(loss.item()) * len(labels)
            count += len(labels)
            steps += 1
        if not steps:
            raise ValueError("No training batches; lower batch_size or provide more samples")
        valid_mse, _, audit = evaluate(model, valid_loader, valid_size, device, False)
        improved = valid_mse <= best_mse
        if improved:
            best_mse, best_epoch, bad_epochs = valid_mse, epoch, 0
            state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            temporary = run_dir / "checkpoint.pt.tmp"
            torch.save(state, temporary)
            os.replace(str(temporary), str(checkpoint))
        else:
            bad_epochs += 1
        should_stop = bad_epochs >= config.patience
        if not should_stop:
            for group in optimizer.param_groups:
                group["lr"] = official_type3_lr(config.learning_rate, epoch)
        row = {"epoch": epoch, "train_mse": train_sse / count, "train_rows": count,
               "steps": steps, "validation_mse": valid_mse, "validation_audit": audit,
               "selected": improved, "bad_epochs": bad_epochs, "learning_rate": lr_used,
               "next_learning_rate": float(optimizer.param_groups[0]["lr"]),
               "elapsed_seconds": time.monotonic() - started}
        history.append(row)
        write_json(run_dir / "history.json", history)
        logger.info("epoch=%d train_mse=%.7f valid_mse=%.7f selected=%s", epoch, row["train_mse"], valid_mse, improved)
        if should_stop:
            break
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    selection = {"rule": "minimum complete finite-label validation SSE/count MSE",
                 "ties_replace_checkpoint": True, "selected_epoch": best_epoch,
                 "selected_validation_mse": best_mse, "epochs_completed": len(history),
                 "test_evaluated_each_epoch": False, "test_used_for_selection": False,
                 "checkpoint": "checkpoint.pt", "stop_reason": "early_stopping" if should_stop else "max_epochs"}
    write_json(run_dir / "selection.json", selection)
    return selection


class SyntheticDataset(Dataset):
    """Small deterministic input for wiring tests; not an investment dataset."""
    def __init__(self, seed, days):
        rng = np.random.RandomState(seed)
        self.features = torch.from_numpy(rng.normal(size=(days * 16, 8, 221)).astype(np.float32))
        self.labels = (0.8 * self.features[:, -1, 0] + 0.2 * self.features[:, -2, 1]).clone()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, position):
        return self.features[position], self.labels[position], position

    def __getitems__(self, positions):
        positions = torch.as_tensor(positions, dtype=torch.long)
        return self.features[positions], self.labels[positions], positions


class SyntheticSplit:
    def __init__(self, seed, days, start):
        self.dataset = SyntheticDataset(seed, days)
        self.labels = self.dataset.labels.numpy()
        self.raw_labels = self.labels.copy()
        self.index = pd.MultiIndex.from_product(
            [pd.bdate_range(start, periods=days), ["S%02d" % n for n in range(16)]],
            names=["datetime", "instrument"])
        self.metadata = {"synthetic": True, "rows": len(self.dataset)}


def synthetic_splits():
    return {"train": SyntheticSplit(8101, 8, "2020-01-01"),
            "valid": SyntheticSplit(8102, 3, "2020-02-03"),
            "test": SyntheticSplit(8103, 4, "2020-03-02")}


def environment_record(args):
    packages = {}
    for name in ("torch", "numpy", "pandas", "scipy", "pyqlib", "PyYAML"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    repository = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repository, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {"python": sys.version, "python_executable": sys.executable,
            "platform": platform.platform(), "hostname": socket.gethostname(), "user": getpass.getuser(),
            "pid": os.getpid(), "cwd": os.getcwd(), "packages": packages,
            "device": args.device, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_cuda": torch.version.cuda, "git_commit": commit, "git_dirty": dirty}


def run_seed(seed, splits, model_config, training_config, args, run_dir, config, environment):
    run_dir.mkdir(exist_ok=False)
    logger = logging.getLogger("qtmaster.seed%d" % seed)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(str(run_dir / "train.log"), encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    console = logging.StreamHandler(sys.stdout)
    logger.handlers = [handler, console]
    status = {"status": "RUNNING", "seed": seed, "started_utc": utc_now(), "ended_utc": None,
              "exit_code": None, "pid": os.getpid(), "device": args.device, "smoke_test": args.smoke_test}
    write_json(run_dir / "status.json", status)
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "environment.json", environment)
    write_json(run_dir / "command.json", {"argv": args.canonical_command,
                                          "shell_command": shlex.join(args.canonical_command)})
    started = time.monotonic()
    try:
        set_seed(seed)
        device = torch.device(args.device)
        model = build_model(model_config).to(device)
        logger.info("seed=%d device=%s parameters=%d smoke=%s", seed, device, sum(p.numel() for p in model.parameters()), args.smoke_test)
        train_loader = make_loader(splits["train"], training_config, True, device)
        valid_loader = make_loader(splits["valid"], training_config, False, device)
        selection = train_model(model, train_loader, valid_loader, len(splits["valid"].dataset), training_config, device, run_dir, logger)
        predictions, evaluation = {}, {}
        for stage in ("valid", "test"):
            split = splits[stage]
            loader = make_loader(split, training_config, False, device)
            mse, values, audit = evaluate(model, loader, len(split.dataset), device)
            prediction = pd.Series(values, index=split.index, name="score")
            prediction.to_csv(run_dir / (stage + "_prediction.csv"))
            predictions[stage] = prediction
            evaluation[stage] = {"mse": mse, "audit": audit}
        if args.smoke_test:
            # Do not publish artificial IC/AR numbers alongside actual experiments.
            result = {"seed": seed, "kind": "SYNTHETIC_SMOKE_ONLY", "metrics": {},
                      "synthetic_evaluation": evaluation, "selection": selection,
                      "backtest_status": "SKIPPED_SYNTHETIC"}
        else:
            metrics = {}
            for stage in ("valid", "test"):
                stage_metrics, daily = daily_metrics(predictions[stage], splits[stage].raw_labels)
                daily.to_csv(run_dir / (stage + "_daily_metrics.csv"))
                evaluation[stage]["metrics"] = stage_metrics
                if stage == "test":
                    metrics.update(stage_metrics)
            metrics.update({"AR": None, "IR": None})
            if args.qlib_provider:
                from .backtest import BRIDGE_DATE, run_backtest
                valid_prediction = predictions["valid"]
                bridge = valid_prediction[valid_prediction.index.get_level_values("datetime") == BRIDGE_DATE]
                backtest = run_backtest(bridge, predictions["test"], config["dataset"], args.qlib_provider, run_dir / "backtest")
                metrics.update({"AR": backtest["AR"], "IR": backtest["IR"]})
            else:
                backtest = {"status": "SKIPPED_NO_PROVIDER", "reason": "Pass --qlib-provider to compute AR/IR"}
                logger.info("AR/IR not calculated: no --qlib-provider supplied")
            result = {"seed": seed, "kind": "EXPERIMENT", "dataset": config["dataset"],
                      "metrics": metrics, "evaluation": evaluation, "selection": selection,
                      "backtest_status": backtest["status"], "backtest": backtest}
        result["elapsed_seconds"] = time.monotonic() - started
        write_json(run_dir / "result.json", result)
        status.update({"status": "DONE", "exit_code": 0})
        logger.info("seed=%d DONE", seed)
        return result
    except BaseException as exc:
        status.update({"status": "FAILED", "exit_code": 130 if isinstance(exc, KeyboardInterrupt) else 1,
                       "error_type": type(exc).__name__, "error": str(exc)})
        logger.exception("Run failed")
        raise
    finally:
        status["ended_utc"] = utc_now()
        status["elapsed_seconds"] = time.monotonic() - started
        write_json(run_dir / "status.json", status)
        handler.close()
        console.close()
        logger.handlers = []


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/csi300.yaml"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--device", default="cpu", help="cpu or e.g. cuda:0")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--trust-pickle", action="store_true", help="Only enable for trusted dataset pickle files")
    parser.add_argument("--qlib-provider", type=Path, help="Optional Qlib cn_data provider for AR/IR")
    parser.add_argument("--smoke-test", action="store_true", help="Two tiny CPU epochs on synthetic data; no financial metrics")
    return parser.parse_args(argv)


def main(argv=None):
    argument_list = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argument_list)
    args.canonical_command = [sys.executable, "-m", "qtmaster.train"] + argument_list
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 or s >= 2 ** 32 for s in args.seeds):
        raise ValueError("Seeds must be distinct integers in [0, 2**32)")
    if args.smoke_test:
        args.device = "cpu"
        torch.set_num_threads(2)
    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config.get("dataset") not in ("csi300", "csi800_direct"):
        raise ValueError("dataset must be csi300 or csi800_direct")
    model_config = ModelConfig(**config.get("model", {}))
    training_values = dict(config.get("training", {}))
    if args.smoke_test:
        training_values.update({"max_epochs": 2, "patience": 2, "num_workers": 0, "batch_size": 32, "eval_batch_size": 32})
    training = TrainingConfig(**training_values)
    if training.lr_schedule != "type3" or training.learning_rate <= 0:
        raise ValueError("Expected positive learning rate and type3 schedule")
    if min(training.batch_size, training.eval_batch_size, training.max_epochs, training.patience) < 1 or training.num_workers < 0:
        raise ValueError("Invalid positive training sizes/worker count")
    config = {"dataset": config["dataset"], "model": asdict(model_config), "training": asdict(training),
              "smoke_test": args.smoke_test, "architecture": "MarketGate -> LLA -> PureMLP -> linear readout",
              "cycle_index": "unused in PureMLP; no provider calendar needed for model forward",
              "test_used_for_selection": False, "checkpoint_ties_replace": True}
    if torch.device(args.device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; choose --device cpu or install compatible PyTorch")
    name = "%s_%s_%s" % ("smoke" if args.smoke_test else config["dataset"], datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"), uuid.uuid4().hex[:8])
    experiment = args.output_root.resolve() / name
    experiment.mkdir(parents=True, exist_ok=False)
    print("Run directory: %s" % experiment, flush=True)
    environment = environment_record(args)
    manifest = {"status": "RUNNING", "started_utc": utc_now(), "ended_utc": None, "exit_code": None,
                "seeds": args.seeds, "pid": os.getpid(), "environment": environment, "config": config,
                "argv": args.canonical_command, "command": shlex.join(args.canonical_command)}
    write_json(experiment / "run.json", manifest)
    try:
        if args.smoke_test:
            splits = synthetic_splits()
        else:
            from .data import load_splits
            splits = load_splits(args.data_root, config["dataset"], trust_pickle=args.trust_pickle)
        write_json(experiment / "data_metadata.json", {stage: split.metadata for stage, split in splits.items()})
        results = []
        for seed in args.seeds:
            results.append(run_seed(seed, splits, model_config, training, args, experiment / ("seed_%d" % seed), config, environment))
        summary = {"kind": "SYNTHETIC_SMOKE_ONLY" if args.smoke_test else "EXPERIMENT",
                   "dataset": config["dataset"], "seeds": args.seeds, "completed_seeds": len(results),
                   "metrics": {} if args.smoke_test else summarize_seeds(results)}
        write_json(experiment / "summary.json", summary)
        if not args.smoke_test:
            pd.DataFrame([{"seed": row["seed"], **row["metrics"]} for row in results]).to_csv(experiment / "per_seed.csv", index=False)
        manifest.update({"status": "DONE", "exit_code": 0})
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    except BaseException as exc:
        manifest.update({"status": "FAILED", "exit_code": 130 if isinstance(exc, KeyboardInterrupt) else 1,
                         "error_type": type(exc).__name__, "error": str(exc)})
        raise
    finally:
        manifest["ended_utc"] = utc_now()
        write_json(experiment / "run.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
