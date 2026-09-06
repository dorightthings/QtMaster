"""Daily cross-sectional metrics, using the existing mainline definitions."""

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


METRIC_NAMES = ("IC", "ICIR", "RankIC", "RankICIR", "AR", "IR")


def daily_metrics(predictions: pd.Series, raw_labels: np.ndarray) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Mean daily correlations and mean/population-std ratios (not annualized)."""
    if len(predictions) != len(raw_labels):
        raise ValueError("Prediction/label length mismatch")
    if not isinstance(predictions.index, pd.MultiIndex) or list(predictions.index.names) != ["datetime", "instrument"]:
        raise ValueError("Expected a datetime,instrument MultiIndex")
    if predictions.index.has_duplicates:
        raise ValueError("Duplicate prediction keys")
    frame = pd.DataFrame({"score": predictions, "label": np.asarray(raw_labels)})
    if not np.isfinite(frame.score.to_numpy()).all():
        raise ValueError("Predictions must all be finite")
    if np.isinf(frame.label.to_numpy()).any():
        raise ValueError("Infinite labels are not supported")
    rows = []
    for date, group in frame.groupby(level="datetime", sort=False):
        valid = group.dropna()
        if len(valid) < 2:
            raise ValueError("Fewer than two finite labels on %s" % date)
        ic = float(valid.score.corr(valid.label))
        rank_ic = float(valid.score.corr(valid.label, method="spearman"))
        if not np.isfinite([ic, rank_ic]).all():
            raise ValueError("Undefined daily correlation on %s (constant inputs?)" % date)
        rows.append({"datetime": date, "IC": ic, "RankIC": rank_ic, "valid_labels": len(valid)})
    if not rows:
        raise ValueError("Empty prediction series")
    daily = pd.DataFrame(rows).set_index("datetime")
    result = {"calendar_days": len(daily), "daily_std_ddof": 0}
    for name, ratio_name in (("IC", "ICIR"), ("RankIC", "RankICIR")):
        values = daily[name].to_numpy(dtype=np.float64)
        mean, std = float(values.mean()), float(values.std(ddof=0))
        if std == 0:
            raise ValueError("Daily %s has zero standard deviation" % name)
        result[name], result[ratio_name] = mean, mean / std
    return result, daily


def summarize_seeds(results):
    """Never turn an unavailable metric (e.g. skipped backtest) into zero."""
    summary = {}
    for metric in METRIC_NAMES:
        values = [row.get("metrics", {}).get(metric) for row in results]
        finite = [float(v) for v in values if v is not None and np.isfinite(v)]
        complete = len(finite) == len(results) and bool(results)
        summary[metric] = {
            "mean": float(np.mean(finite)) if complete else None,
            "std": float(np.std(finite, ddof=0)) if complete else None,
            "std_ddof": 0,
            "available_seeds": len(finite),
            "requested_seeds": len(results),
        }
    return summary
