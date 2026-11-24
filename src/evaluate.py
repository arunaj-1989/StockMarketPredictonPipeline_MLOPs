"""Forecast evaluation metrics and promotion gates (pure functions, no I/O)."""

from __future__ import annotations

import numpy as np
import pandas as pd

QUANTILES = (0.10, 0.50, 0.90)
MIN_GAIN = 0.0  # a challenger must beat the comparison model by strictly more than this


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    diff = np.asarray(y_true, dtype="float64") - np.asarray(y_pred, dtype="float64")
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1) * diff)))


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Share of rows where the predicted sign matches the realised sign (zeros excluded)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = (y_true != 0) & (y_pred != 0)
    if not mask.any():
        return float("nan")
    return float(np.mean(np.sign(y_true[mask]) == np.sign(y_pred[mask])))


def interval_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def enforce_quantile_order(q: np.ndarray) -> np.ndarray:
    """Sort quantile columns per row so that q10 <= q50 <= q90 (monotone rearrangement)."""
    return np.sort(np.asarray(q, dtype="float64"), axis=1)


def persistence_baseline(n: int) -> np.ndarray:
    """Naive persistence P_hat(t+k) = P_t, i.e. a predicted log return of exactly zero."""
    return np.zeros(n)


def horizon_metrics(
    y_true: np.ndarray, quantile_pred: np.ndarray, momentum_sign: np.ndarray | None = None
) -> dict[str, float]:
    """Metrics for one horizon. ``quantile_pred`` has columns [q10, q50, q90]."""
    q = enforce_quantile_order(quantile_pred)
    metrics = {
        "pinball_q10": pinball_loss(y_true, q[:, 0], 0.10),
        "pinball_q50": pinball_loss(y_true, q[:, 1], 0.50),
        "pinball_q90": pinball_loss(y_true, q[:, 2], 0.90),
        "directional_accuracy": directional_accuracy(y_true, q[:, 1]),
        "coverage_q10_q90": interval_coverage(y_true, q[:, 0], q[:, 2]),
    }
    metrics["pinball_mean"] = float(np.mean([metrics[f"pinball_q{p}"] for p in (10, 50, 90)]))
    zero = persistence_baseline(len(y_true))
    metrics["persistence_pinball_q50"] = pinball_loss(y_true, zero, 0.50)
    if momentum_sign is not None:
        metrics["persistence_directional_accuracy"] = directional_accuracy(y_true, momentum_sign)
    return metrics


def point_metrics(
    y_true: np.ndarray, point_pred: np.ndarray, momentum_sign: np.ndarray | None = None
) -> dict[str, float]:
    """Metrics for a point (median) forecaster such as the Ridge baseline."""
    metrics = {
        "pinball_q50": pinball_loss(y_true, point_pred, 0.50),
        "directional_accuracy": directional_accuracy(y_true, point_pred),
        "persistence_pinball_q50": pinball_loss(y_true, persistence_baseline(len(y_true)), 0.50),
    }
    if momentum_sign is not None:
        metrics["persistence_directional_accuracy"] = directional_accuracy(y_true, momentum_sign)
    return metrics


def aggregate(per_horizon: dict[int, dict[str, float]]) -> dict[str, float]:
    """Mean of every metric across horizons."""
    frame = pd.DataFrame(per_horizon).T
    return {col: float(frame[col].mean()) for col in frame.columns}


def persistence_lag_check(
    y_true: np.ndarray, median_pred: np.ndarray, lagged_return: np.ndarray
) -> dict[str, float | bool]:
    """Detect a model that merely echoes the current/lagged return.

    The model must (a) have lower q50 pinball loss than persistence (zero return) and
    (b) not be better explained by the lagged return itself: its correlation with the
    realised target must exceed that of the naive "repeat the last return" forecast.
    """
    y_true = np.asarray(y_true)
    median_pred = np.asarray(median_pred)
    lagged_return = np.asarray(lagged_return)
    model_loss = pinball_loss(y_true, median_pred, 0.5)
    persistence_loss = pinball_loss(y_true, persistence_baseline(len(y_true)), 0.5)

    def _corr(a, b):
        if np.std(a) == 0 or np.std(b) == 0:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    model_corr = _corr(median_pred, y_true)
    lag_corr = _corr(lagged_return, y_true)
    echo_corr = _corr(median_pred, lagged_return)
    return {
        "model_pinball_q50": model_loss,
        "persistence_pinball_q50": persistence_loss,
        "model_target_corr": model_corr,
        "lagged_return_target_corr": lag_corr,
        "model_lagged_return_corr": echo_corr,
        "beats_persistence": bool(model_loss < persistence_loss - MIN_GAIN),
        "not_a_lag_echo": bool(model_corr > abs(lag_corr)),
    }


def challenger_vs_champion_gate(
    challenger: dict[str, float],
    champion: dict[str, float] | None,
    min_relative_gain: float = 0.0,
) -> dict[str, object]:
    """Holdout gate: promote only if the challenger improves on the champion.

    Compares mean pinball loss (lower is better) and requires directional accuracy not to
    degrade. With no champion the challenger passes by default (the baseline checks still
    apply in ``baseline_gate``).
    """
    if champion is None:
        return {"promote": True, "reason": "no existing champion"}
    c_loss, p_loss = challenger["pinball_mean"], champion["pinball_mean"]
    gain = (p_loss - c_loss) / p_loss if p_loss else 0.0
    loss_ok = gain > min_relative_gain
    da_ok = challenger["directional_accuracy"] >= champion["directional_accuracy"] - 1e-12
    promote = bool(loss_ok and da_ok)
    return {
        "promote": promote,
        "reason": (
            f"pinball_mean {c_loss:.6f} vs {p_loss:.6f} (gain {gain:+.2%}); "
            f"DA {challenger['directional_accuracy']:.4f} vs {champion['directional_accuracy']:.4f}"
        ),
        "relative_gain": float(gain),
        "loss_ok": bool(loss_ok),
        "directional_ok": bool(da_ok),
    }


def baseline_gate(v2: dict[str, float], v1: dict[str, float]) -> dict[str, object]:
    """V2 must beat both the V1 Ridge baseline and naive persistence on q50 pinball loss."""
    beats_v1 = v2["pinball_q50"] < v1["pinball_q50"] - MIN_GAIN
    beats_persistence = v2["pinball_q50"] < v2["persistence_pinball_q50"] - MIN_GAIN
    return {
        "v2_pinball_q50": v2["pinball_q50"],
        "v1_pinball_q50": v1["pinball_q50"],
        "persistence_pinball_q50": v2["persistence_pinball_q50"],
        "beats_v1": bool(beats_v1),
        "beats_persistence": bool(beats_persistence),
        "passed": bool(beats_v1 and beats_persistence),
    }


# ----------------------------------------------------------------- cross-sectional ranking
IC_MIN = 0.03  # mean daily rank correlation needed out of sample
IC_T_MIN = 3.0  # t-statistic on non-overlapping dates
IC_POSITIVE_FOLDS_MIN = 0.6  # share of walk-forward folds with a positive mean IC
ROUND_TRIP_COST = 0.005  # ~50 bps NSE STT + brokerage, the same hurdle used for trades


def daily_rank_ic(df: pd.DataFrame, score: str, target: str) -> pd.Series:
    """Spearman correlation between score and realised target across stocks, for every date."""
    d = df[["date", score, target]].dropna()
    ranked = d.assign(_s=d.groupby("date")[score].rank(), _t=d.groupby("date")[target].rank())
    return ranked.groupby("date").apply(lambda g: g["_s"].corr(g["_t"]), include_groups=False).dropna()


def ic_summary(ic: pd.Series, horizon: int) -> dict[str, float]:
    """Mean IC and its t-statistic. Targets overlap for horizon > 1, so t uses every horizon-th date."""
    if len(ic) < 2:
        return {"n_days": len(ic), "mean_ic": float("nan"), "ic_std": float("nan"),
                "t_stat": float("nan"), "positive_share": float("nan")}
    sparse = ic.iloc[::horizon]
    std = float(sparse.std())
    return {
        "n_days": len(ic),
        "mean_ic": float(ic.mean()),
        "ic_std": float(ic.std()),
        "t_stat": float(sparse.mean() / std * np.sqrt(len(sparse))) if std > 0 else float("nan"),
        "positive_share": float((ic > 0).mean()),
    }


def quintile_alpha(df: pd.DataFrame, score: str, target: str) -> dict[str, float]:
    """Average market-relative return of the top and bottom quintile by score (log-return units)."""
    d = df[["date", score, target]].dropna().copy()
    d["_q"] = d.groupby("date")[score].rank(method="first", pct=True)
    top = d[d["_q"] > 0.8].groupby("date")[target].mean()
    bottom = d[d["_q"] <= 0.2].groupby("date")[target].mean()
    return {
        "top_quintile_alpha": float(top.mean()),
        "bottom_quintile_alpha": float(bottom.mean()),
        "long_short_spread": float((top - bottom).dropna().mean()),
    }


def walk_forward_windows(
    n_dates: int, n_folds: int = 5, first_test_frac: float = 0.55, embargo: int = 10
) -> list[dict[str, int]]:
    """Expanding-window folds over date positions. Each fold trains on everything up to
    ``embargo`` dates before its test block, so no label window crosses into the test period."""
    start = int(n_dates * first_test_frac)
    length = (n_dates - start) // n_folds
    folds = []
    for i in range(n_folds):
        test_start = start + i * length
        test_end = n_dates if i == n_folds - 1 else test_start + length
        folds.append({"train_end": test_start - embargo, "test_start": test_start, "test_end": test_end})
    return folds


def ranking_gate(summary: dict[str, float], fold_mean_ics: list[float]) -> dict[str, object]:
    """A ranker may drive signals only if its out-of-sample IC is large, significant and consistent."""
    positive = float(np.mean([ic > 0 for ic in fold_mean_ics])) if fold_mean_ics else 0.0
    checks = {
        "mean_ic": bool(summary["mean_ic"] >= IC_MIN),
        "t_stat": bool(summary["t_stat"] >= IC_T_MIN),
        "positive_folds": bool(positive >= IC_POSITIVE_FOLDS_MIN),
    }
    return {
        "passed": all(checks.values()), "checks": checks, "positive_fold_share": positive,
        "thresholds": {"mean_ic": IC_MIN, "t_stat": IC_T_MIN, "positive_folds": IC_POSITIVE_FOLDS_MIN},
    }
