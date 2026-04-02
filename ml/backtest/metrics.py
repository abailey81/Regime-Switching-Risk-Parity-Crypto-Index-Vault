"""
Performance Metrics Computation.

Provides all standard quantitative portfolio metrics used in the
walk-forward backtest and Monte Carlo stress testing.

Annualisation note:
  - Hourly data: x sqrt(8760) for volatility, x 8760 for returns
  - 8760 = 365 * 24 (hours per year)
"""
import numpy as np
import pandas as pd
from scipy.stats import norm


def annualised_return(returns: np.ndarray, periods_per_year: float = 8760) -> float:
    """Annualised geometric return from a series of periodic log returns."""
    total = np.sum(returns)
    n_periods = len(returns)
    if n_periods == 0:
        return 0.0
    return (np.exp(total * periods_per_year / n_periods) - 1)


def annualised_volatility(returns: np.ndarray, periods_per_year: float = 8760) -> float:
    """Annualised volatility (standard deviation of returns)."""
    if len(returns) < 2:
        return 0.0
    return np.std(returns, ddof=1) * np.sqrt(periods_per_year)


def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.045,
                 periods_per_year: float = 8760) -> float:
    """
    Annualised Sharpe ratio.
    SR = (annualised_return - risk_free_rate) / annualised_volatility
    """
    ann_ret = annualised_return(returns, periods_per_year)
    ann_vol = annualised_volatility(returns, periods_per_year)
    if ann_vol < 1e-10:
        return 0.0
    return (ann_ret - risk_free_rate) / ann_vol


def sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.045,
                  periods_per_year: float = 8760) -> float:
    """
    Annualised Sortino ratio (penalises downside volatility only).
    Uses downside deviation instead of total standard deviation.
    """
    ann_ret = annualised_return(returns, periods_per_year)
    rf_per_period = risk_free_rate / periods_per_year
    downside = returns[returns < rf_per_period] - rf_per_period
    if len(downside) < 2:
        return 0.0
    downside_vol = np.std(downside, ddof=1) * np.sqrt(periods_per_year)
    if downside_vol < 1e-10:
        return 0.0
    return (ann_ret - risk_free_rate) / downside_vol


def maximum_drawdown(returns: np.ndarray) -> float:
    """Maximum drawdown from cumulative return series."""
    cum_returns = np.exp(np.cumsum(returns))
    running_max = np.maximum.accumulate(cum_returns)
    drawdowns = cum_returns / running_max - 1.0
    return np.min(drawdowns)


def drawdown_series(returns: np.ndarray) -> np.ndarray:
    """Full drawdown time series."""
    cum_returns = np.exp(np.cumsum(returns))
    running_max = np.maximum.accumulate(cum_returns)
    return cum_returns / running_max - 1.0


def calmar_ratio(returns: np.ndarray, periods_per_year: float = 8760) -> float:
    """Calmar ratio = annualised return / |max drawdown|."""
    ann_ret = annualised_return(returns, periods_per_year)
    mdd = abs(maximum_drawdown(returns))
    if mdd < 1e-10:
        return 0.0
    return ann_ret / mdd


def cvar(returns: np.ndarray, confidence: float = 0.05) -> float:
    """
    Historical Conditional Value-at-Risk (Expected Shortfall) at given confidence.
    CVaR_5% = average of worst 5% of returns.
    Returns a positive number (the expected loss magnitude).
    """
    if len(returns) < 20:
        return 0.0
    sorted_returns = np.sort(returns)
    n_tail = max(1, int(confidence * len(sorted_returns)))
    return -np.mean(sorted_returns[:n_tail])


def var(returns: np.ndarray, confidence: float = 0.05) -> float:
    """Historical Value-at-Risk at given confidence level."""
    if len(returns) < 20:
        return 0.0
    return -np.percentile(returns, confidence * 100)


def average_turnover(weight_changes: list) -> float:
    """Average L1 turnover per rebalance event."""
    if not weight_changes:
        return 0.0
    turnovers = [np.abs(wc).sum() for wc in weight_changes]
    return np.mean(turnovers)


def tracking_error(returns: np.ndarray, benchmark_returns: np.ndarray,
                   periods_per_year: float = 8760) -> float:
    """Annualised tracking error vs benchmark."""
    min_len = min(len(returns), len(benchmark_returns))
    diff = returns[:min_len] - benchmark_returns[:min_len]
    return np.std(diff, ddof=1) * np.sqrt(periods_per_year)


def information_ratio(returns: np.ndarray, benchmark_returns: np.ndarray,
                      periods_per_year: float = 8760) -> float:
    """Information ratio = excess return / tracking error."""
    te = tracking_error(returns, benchmark_returns, periods_per_year)
    if te < 1e-10:
        return 0.0
    excess = annualised_return(returns, periods_per_year) - annualised_return(benchmark_returns, periods_per_year)
    return excess / te


# ═══════════════════════════════════════════════════════════════
# EXTENDED METRICS (from stat-arb advanced_metrics.py)
# ═══════════════════════════════════════════════════════════════

def omega_ratio(returns: np.ndarray, threshold: float = 0.0) -> float:
    """
    Omega ratio: sum of gains above threshold / sum of losses below threshold.
    Omega > 1 indicates positive expectancy. From stat-arb advanced_metrics.py.
    """
    gains = np.sum(np.maximum(returns - threshold, 0))
    losses = np.sum(np.maximum(threshold - returns, 0))
    if losses < 1e-12:
        return float('inf') if gains > 0 else 1.0
    return float(gains / losses)


def ulcer_index(returns: np.ndarray) -> float:
    """
    Ulcer Index: RMS of percentage drawdowns. Lower = less painful.
    Measures both depth and duration of drawdowns.
    From stat-arb advanced_metrics.py.
    """
    cum = np.exp(np.cumsum(returns))
    peak = np.maximum.accumulate(cum)
    dd_pct = ((cum - peak) / peak) * 100
    return float(np.sqrt(np.mean(dd_pct ** 2)))


def pain_index(returns: np.ndarray) -> float:
    """
    Pain Index: mean absolute drawdown. From stat-arb advanced_metrics.py.
    """
    cum = np.exp(np.cumsum(returns))
    peak = np.maximum.accumulate(cum)
    dd = np.abs((cum - peak) / peak)
    return float(np.mean(dd))


def burke_ratio(returns: np.ndarray, periods_per_year: float = 8760) -> float:
    """
    Burke ratio: annualised return / sqrt(sum of squared drawdowns).
    Penalises multiple drawdowns more than a single deep one.
    """
    ann_ret = annualised_return(returns, periods_per_year)
    cum = np.exp(np.cumsum(returns))
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    dd_squared_sum = np.sum(dd[dd < 0] ** 2)
    if dd_squared_sum < 1e-12:
        return 0.0
    return float(ann_ret / np.sqrt(dd_squared_sum / len(returns)))


def tail_ratio(returns: np.ndarray) -> float:
    """
    Tail ratio: |95th percentile| / |5th percentile|.
    Values > 1 indicate fatter right tail (more upside than downside).
    """
    p95 = np.percentile(returns, 95)
    p5 = np.percentile(returns, 5)
    if abs(p5) < 1e-12:
        return 0.0
    return float(abs(p95) / abs(p5))


def gain_to_pain_ratio(returns: np.ndarray) -> float:
    """
    Gain-to-Pain ratio: sum of all returns / sum of absolute negative returns.
    Also known as the "pain ratio". GPR > 1 = gains exceed pain.
    """
    total = np.sum(returns)
    pain = np.sum(np.abs(returns[returns < 0]))
    if pain < 1e-12:
        return float('inf') if total > 0 else 0.0
    return float(total / pain)


def rachev_ratio(returns: np.ndarray, alpha: float = 0.05) -> float:
    """
    Rachev ratio: CVaR of gains (right tail) / CVaR of losses (left tail).
    Measures the ratio of expected best-case vs expected worst-case.
    Values > 1 indicate favourable tail asymmetry.
    """
    if len(returns) < 20:
        return 0.0
    sorted_ret = np.sort(returns)
    n_tail = max(1, int(alpha * len(sorted_ret)))
    left_cvar = -np.mean(sorted_ret[:n_tail])
    right_cvar = np.mean(sorted_ret[-n_tail:])
    if left_cvar < 1e-12:
        return 0.0
    return float(right_cvar / left_cvar)


def sterling_ratio(returns: np.ndarray, periods_per_year: float = 8760) -> float:
    """
    Sterling ratio: annualised return / (|max drawdown| - 10%).
    A stricter version of Calmar that subtracts a 10% buffer from the
    denominator, penalising strategies with marginal drawdowns.
    """
    ann_ret = annualised_return(returns, periods_per_year)
    mdd = abs(maximum_drawdown(returns))
    denom = mdd - 0.10
    if denom < 1e-10:
        return 0.0
    return float(ann_ret / denom)


def kappa_ratio(returns: np.ndarray, order: float = 3.0,
                threshold: float = 0.0, periods_per_year: float = 8760) -> float:
    """
    Kappa ratio of given order (generalisation of Omega and Sortino).
    Kappa_n = (annualised_return - threshold) / LPM_n^(1/n)
    where LPM_n = E[max(threshold - r, 0)^n].
    Order 1 = Omega - 1, Order 2 = Sortino.
    """
    ann_ret = annualised_return(returns, periods_per_year)
    shortfall = np.maximum(threshold - returns, 0)
    lpm = np.mean(shortfall ** order)
    if lpm < 1e-12:
        return 0.0
    lpm_root = lpm ** (1.0 / order) * np.sqrt(periods_per_year)
    if lpm_root < 1e-12:
        return 0.0
    return float((ann_ret - threshold) / lpm_root)


def rolling_sharpe_series(returns: np.ndarray, window: int = 2160,
                          periods_per_year: float = 8760) -> np.ndarray:
    """Rolling Sharpe ratio time series (default: 90-day window)."""
    result = np.full(len(returns), np.nan)
    for i in range(window, len(returns)):
        chunk = returns[i - window:i]
        mu = np.mean(chunk) * periods_per_year
        sig = np.std(chunk, ddof=1) * np.sqrt(periods_per_year)
        result[i] = mu / sig if sig > 1e-12 else 0.0
    return result


# ═══════════════════════════════════════════════════════════════
# BOOTSTRAP CONFIDENCE INTERVALS
# ═══════════════════════════════════════════════════════════════

def bootstrap_metric(returns: np.ndarray, metric_func, n_resamples: int = 1000,
                     confidence: float = 0.95, seed: int = 42) -> dict:
    """
    Compute bootstrap confidence interval for a given metric function.

    Args:
        returns: Array of periodic log returns
        metric_func: Function that takes returns array and returns a scalar
        n_resamples: Number of bootstrap resamples
        confidence: Confidence level (e.g. 0.95 for 95% CI)
        seed: Random seed

    Returns:
        Dict with 'point_estimate', 'ci_lower', 'ci_upper', 'std_error'
    """
    rng = np.random.RandomState(seed)
    n = len(returns)
    if n < 20:
        val = metric_func(returns)
        return {"point_estimate": val, "ci_lower": val, "ci_upper": val, "std_error": 0.0}

    point_estimate = metric_func(returns)
    bootstrap_estimates = np.zeros(n_resamples)

    for i in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        bootstrap_estimates[i] = metric_func(returns[idx])

    alpha = (1 - confidence) / 2
    ci_lower = float(np.percentile(bootstrap_estimates, alpha * 100))
    ci_upper = float(np.percentile(bootstrap_estimates, (1 - alpha) * 100))
    std_error = float(np.std(bootstrap_estimates))

    return {
        "point_estimate": float(point_estimate),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "std_error": std_error,
    }


def paired_bootstrap_test(returns_a: np.ndarray, returns_b: np.ndarray,
                          metric_func, n_resamples: int = 1000,
                          seed: int = 42) -> dict:
    """
    Paired bootstrap test for whether metric(A) > metric(B) is significant.

    Tests H0: metric(A) <= metric(B) vs H1: metric(A) > metric(B).

    Args:
        returns_a: Returns for strategy A (ensemble)
        returns_b: Returns for strategy B (benchmark)
        metric_func: Function that takes returns and returns a scalar
        n_resamples: Number of bootstrap resamples
        seed: Random seed

    Returns:
        Dict with 'metric_a', 'metric_b', 'diff', 'p_value', 'significant_5pct'
    """
    rng = np.random.RandomState(seed)
    min_len = min(len(returns_a), len(returns_b))
    a = returns_a[:min_len]
    b = returns_b[:min_len]
    n = len(a)

    metric_a = metric_func(a)
    metric_b = metric_func(b)
    observed_diff = metric_a - metric_b

    bootstrap_diffs = np.zeros(n_resamples)
    for i in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        bootstrap_diffs[i] = metric_func(a[idx]) - metric_func(b[idx])

    # p-value: proportion of bootstrap diffs <= 0 (one-sided test)
    p_value = float(np.mean(bootstrap_diffs <= 0))

    return {
        "metric_a": float(metric_a),
        "metric_b": float(metric_b),
        "diff": float(observed_diff),
        "p_value": p_value,
        "significant_5pct": p_value < 0.05,
    }


# ═══════════════════════════════════════════════════════════════
# COMPREHENSIVE METRICS (single definition)
# ═══════════════════════════════════════════════════════════════

def compute_all_metrics(returns: np.ndarray, benchmark_returns: np.ndarray = None,
                        weight_changes: list = None,
                        periods_per_year: float = 8760,
                        risk_free_rate: float = 0.045) -> dict:
    """
    Compute ALL metrics for a return series (21+ metrics).

    Annualisation: hourly data uses periods_per_year=8760.
      - Volatility: x sqrt(8760)
      - Returns: x 8760

    Args:
        returns: Array of periodic log returns
        benchmark_returns: Optional benchmark for tracking error / IR
        weight_changes: Optional list of weight change vectors for turnover
        periods_per_year: Annualisation factor (8760 for hourly)
        risk_free_rate: Annual risk-free rate

    Returns:
        Dictionary of all metrics
    """
    metrics = {
        # Core performance
        "annualised_return": annualised_return(returns, periods_per_year),
        "annualised_volatility": annualised_volatility(returns, periods_per_year),
        "sharpe_ratio": sharpe_ratio(returns, risk_free_rate, periods_per_year),
        "sortino_ratio": sortino_ratio(returns, risk_free_rate, periods_per_year),
        "max_drawdown": maximum_drawdown(returns),
        "calmar_ratio": calmar_ratio(returns, periods_per_year),
        "total_return": float(np.exp(np.sum(returns)) - 1),

        # Tail risk
        "cvar_5pct": cvar(returns, 0.05),
        "var_5pct": var(returns, 0.05),

        # Extended risk-adjusted ratios
        "omega_ratio": omega_ratio(returns),
        "ulcer_index": ulcer_index(returns),
        "pain_index": pain_index(returns),
        "burke_ratio": burke_ratio(returns, periods_per_year),
        "tail_ratio": tail_ratio(returns),
        "gain_to_pain": gain_to_pain_ratio(returns),
        "rachev_ratio": rachev_ratio(returns),
        "sterling_ratio": sterling_ratio(returns, periods_per_year),
        "kappa_3": kappa_ratio(returns, order=3.0, periods_per_year=periods_per_year),

        # Distribution shape
        "skewness": float(pd.Series(returns).skew()),
        "kurtosis": float(pd.Series(returns).kurtosis()),

        # Metadata
        "n_observations": len(returns),
    }

    if weight_changes:
        metrics["avg_turnover"] = average_turnover(weight_changes)

    if benchmark_returns is not None:
        metrics["tracking_error"] = tracking_error(returns, benchmark_returns, periods_per_year)
        metrics["information_ratio"] = information_ratio(returns, benchmark_returns, periods_per_year)

    return metrics


def format_metrics_table(metrics_dict: dict, name: str = "Strategy") -> pd.DataFrame:
    """Format metrics as a display-ready DataFrame."""
    formatted = {}
    for key, val in metrics_dict.items():
        if "return" in key or "volatility" in key or "error" in key or "drawdown" in key or "cvar" in key or "var" in key:
            formatted[key] = f"{val:.2%}"
        elif "ratio" in key:
            formatted[key] = f"{val:.3f}"
        elif "turnover" in key:
            formatted[key] = f"{val:.4f}"
        else:
            formatted[key] = f"{val}"

    return pd.DataFrame(formatted, index=[name]).T
