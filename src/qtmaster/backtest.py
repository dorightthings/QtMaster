"""Optional Qlib 0.9.7 Top30/Drop30 backtest for the original short datasets.

The headline AR/IR use daily portfolio-minus-benchmark returns without costs.
Transaction costs are still present in the simulator and saved in the report.
This module does not import Qlib until a backtest is explicitly requested.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


CONTRACTS = {
    "csi300": {"benchmark": "SH000300", "bridge_rows": 300, "test_rows": 183300},
    "csi800_direct": {"benchmark": "SH000906", "bridge_rows": 800, "test_rows": 488721},
}
BRIDGE_DATE = pd.Timestamp("2020-06-30")
TEST_START = pd.Timestamp("2020-07-01")
TEST_END = pd.Timestamp("2022-12-30")


def validate_signal(bridge, test, dataset, provider_uri):
    """Refuse incomplete dates/rows rather than silently changing the backtest."""
    contract = CONTRACTS[dataset]
    for role, prediction in (("bridge", bridge), ("test", test)):
        if not isinstance(prediction, pd.Series):
            raise TypeError("%s predictions must be a Series" % role)
        index = prediction.index
        if not isinstance(index, pd.MultiIndex) or list(index.names) != ["datetime", "instrument"]:
            raise ValueError("Expected datetime,instrument MultiIndex")
        if index.has_duplicates or not index.is_monotonic_increasing:
            raise ValueError("Predictions must have unique, sorted keys")
        if not np.isfinite(prediction.to_numpy(dtype=np.float64)).all():
            raise ValueError("Predictions must be finite")
        dates = pd.DatetimeIndex(index.get_level_values("datetime"))
        if dates.tz is not None or not dates.equals(dates.normalize()):
            raise ValueError("Prediction dates must be timezone-naive trading dates")
        if len(prediction) != contract[role + "_rows"]:
            raise ValueError("%s rows do not match %s short-data contract" % (role, dataset))
    bridge_dates = pd.DatetimeIndex(bridge.index.get_level_values("datetime").unique())
    test_dates = pd.DatetimeIndex(test.index.get_level_values("datetime").unique())
    if list(bridge_dates) != [BRIDGE_DATE]:
        raise ValueError("Backtest requires selected-model predictions for 2020-06-30")
    if len(test_dates) != 611 or test_dates[0] != TEST_START or test_dates[-1] != TEST_END:
        raise ValueError("Backtest requires all 611 test trading dates")
    calendar_path = Path(provider_uri) / "calendars" / "day.txt"
    if not calendar_path.is_file():
        raise FileNotFoundError("Missing Qlib calendar: %s" % calendar_path)
    calendar = pd.DatetimeIndex(pd.to_datetime(calendar_path.read_text().splitlines()))
    if calendar.has_duplicates or not calendar.is_monotonic_increasing:
        raise ValueError("Provider calendar must be unique and sorted")
    expected = calendar[(calendar >= TEST_START) & (calendar <= TEST_END)]
    if not test_dates.equals(expected) or BRIDGE_DATE not in calendar:
        raise ValueError("Prediction trading dates do not match the provider calendar")
    signal = pd.concat([bridge.rename("score"), test.rename("score")])
    if signal.index.has_duplicates or not signal.index.is_monotonic_increasing:
        raise ValueError("Bridge and test signals overlap or are unsorted")
    return signal, expected


def run_backtest(bridge_prediction, test_prediction, dataset, provider_uri, output_dir):
    """Run once on the validation-selected checkpoint; never tune on this result."""
    provider_uri, output_dir = Path(provider_uri).resolve(), Path(output_dir).resolve()
    signal, expected_days = validate_signal(bridge_prediction, test_prediction, dataset, provider_uri)
    output_dir.mkdir(parents=True, exist_ok=False)
    import qlib
    from qlib.backtest import backtest
    from qlib.contrib.evaluate import risk_analysis

    if str(qlib.__version__) != "0.9.7":
        raise RuntimeError("This backtest protocol requires pyqlib==0.9.7")
    qlib.init(provider_uri=str(provider_uri), region="cn")
    exchange = {"deal_price": "close", "open_cost": 0.0015, "close_cost": 0.0025,
                "min_cost": 5.0, "limit_threshold": 0.095, "trade_unit": 100}
    portfolio, _ = backtest(
        start_time="2020-07-01", end_time="2022-12-31", account=100000000,
        benchmark=CONTRACTS[dataset]["benchmark"], exchange_kwargs=exchange,
        strategy={"class": "TopkDropoutStrategy", "module_path": "qlib.contrib.strategy",
                  "kwargs": {"signal": signal, "topk": 30, "n_drop": 30}},
        executor={"class": "SimulatorExecutor", "module_path": "qlib.backtest.executor",
                  "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True}},
    )
    report, positions = portfolio["1day"]
    if not pd.DatetimeIndex(report.index).equals(expected_days):
        raise ValueError("Backtest report does not cover the complete test calendar")
    if not np.isfinite(report[["return", "bench", "cost", "turnover"]].to_numpy()).all():
        raise ValueError("Non-finite backtest report")
    if float(report.iloc[0].turnover) <= 0:
        raise ValueError("First test day did not consume bridge-day predictions")
    risk = risk_analysis(report["return"] - report["bench"], freq="1day")
    result = {"status": "DONE", "AR": float(risk.loc["annualized_return", "risk"]),
              "IR": float(risk.loc["information_ratio", "risk"]),
              "headline": "excess_return_without_cost", "topk": 30, "n_drop": 30,
              "benchmark": CONTRACTS[dataset]["benchmark"], "provider_uri": str(provider_uri),
              "qlib_version": str(qlib.__version__), "exchange": exchange,
              "report_days": len(report), "bridge_date": str(BRIDGE_DATE.date())}
    if not np.isfinite([result["AR"], result["IR"]]).all():
        raise ValueError("Non-finite backtest metrics")
    report.to_csv(output_dir / "report.csv")
    risk.to_csv(output_dir / "risk_analysis.csv")
    signal.to_csv(output_dir / "signal.csv")
    pd.to_pickle(positions, output_dir / "positions.pkl")
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
