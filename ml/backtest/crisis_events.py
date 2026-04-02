"""
Crypto Crisis Event Definitions for Regime-Stratified Backtest Analysis.

Adapted from Crypto-Statistical-Arbitrage/backtesting/analysis/crisis_analyzer.py.
All 12 major crypto crisis events from 2020-2024, used for:
  - Regime-stratified performance metrics (Sharpe during crisis vs bull)
  - Stress testing the ensemble's regime detection capability
  - HMM validation (did the model detect the crisis?)
  - Circuit breaker trigger analysis

Each event includes: dates, type, severity, BTC drawdown, and expected
portfolio behaviour under the ensemble strategy.

Severity classification (based on BTC drawdown):
  - mild:     BTC drawdown < 15%
  - moderate: BTC drawdown 15-30%
  - severe:   BTC drawdown > 30%
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional


@dataclass
class CrisisEvent:
    """Single crypto crisis event definition."""
    name: str
    start: str              # YYYY-MM-DD
    end: str                # YYYY-MM-DD
    crisis_type: str        # market_crash, exchange_failure, protocol_exploit, regulatory, macro
    severity: int           # 1-5 scale
    btc_drawdown_pct: float # Approximate BTC drawdown during event
    description: str
    expected_hmm_regime: str  # What HMM should detect
    expected_ensemble_action: str  # What the ensemble should do

    @property
    def severity_label(self) -> str:
        """Classify severity as mild/moderate/severe based on BTC drawdown."""
        dd = abs(self.btc_drawdown_pct)
        if dd < 15:
            return "mild"
        elif dd < 30:
            return "moderate"
        else:
            return "severe"

    @property
    def start_dt(self) -> datetime:
        return datetime.strptime(self.start, "%Y-%m-%d")

    @property
    def end_dt(self) -> datetime:
        return datetime.strptime(self.end, "%Y-%m-%d")

    @property
    def duration_days(self) -> int:
        return (self.end_dt - self.start_dt).days


# All 12 major crypto crises (from stat-arb walk_forward_optimizer.py)
CRYPTO_CRISES: List[CrisisEvent] = [
    CrisisEvent(
        name="COVID Crash",
        start="2020-03-01", end="2020-04-15",
        crisis_type="macro", severity=5,
        btc_drawdown_pct=-52.0,
        description="Global pandemic selloff. BTC dropped from $9.1K to $4.3K in 48 hours.",
        expected_hmm_regime="crisis",
        expected_ensemble_action="Full defensive: 80%+ stablecoins/Treasuries",
    ),
    CrisisEvent(
        name="May 2021 Crash",
        start="2021-05-10", end="2021-06-30",
        crisis_type="regulatory", severity=4,
        btc_drawdown_pct=-54.0,
        description="China mining ban + Elon Tesla reversal. BTC from $58K to $29K.",
        expected_hmm_regime="crisis",
        expected_ensemble_action="Defensive within 24-48 hours of onset",
    ),
    CrisisEvent(
        name="UST/Luna Collapse",
        start="2022-05-01", end="2022-06-30",
        crisis_type="protocol_exploit", severity=5,
        btc_drawdown_pct=-40.0,
        description="Algorithmic stablecoin death spiral. $60B wiped. Systemic contagion.",
        expected_hmm_regime="crisis",
        expected_ensemble_action="Crisis regime detected within 36 hours. Circuit breaker triggered.",
    ),
    CrisisEvent(
        name="3AC Liquidation",
        start="2022-06-13", end="2022-07-15",
        crisis_type="market_crash", severity=4,
        btc_drawdown_pct=-25.0,
        description="Three Arrows Capital forced liquidation. Cascade across lenders.",
        expected_hmm_regime="crisis",
        expected_ensemble_action="Sustained defensive allocation through contagion",
    ),
    CrisisEvent(
        name="Celsius Bankruptcy",
        start="2022-07-13", end="2022-07-30",
        crisis_type="exchange_failure", severity=3,
        btc_drawdown_pct=-10.0,
        description="Celsius Network halts withdrawals, files bankruptcy.",
        expected_hmm_regime="crisis",
        expected_ensemble_action="Maintain defensive from prior 3AC crisis",
    ),
    CrisisEvent(
        name="FTX Collapse",
        start="2022-11-01", end="2022-12-31",
        crisis_type="exchange_failure", severity=5,
        btc_drawdown_pct=-27.0,
        description="FTX/Alameda fraud revealed. $8B customer funds missing. Systemic crisis.",
        expected_hmm_regime="crisis",
        expected_ensemble_action="Crisis detection within hours. Maximum defensive allocation.",
    ),
    CrisisEvent(
        name="SVB/USDC Depeg",
        start="2023-03-01", end="2023-04-15",
        crisis_type="macro", severity=4,
        btc_drawdown_pct=-10.0,
        description="Silicon Valley Bank collapse. USDC depegged to $0.87. Banking crisis fears.",
        expected_hmm_regime="crisis",
        expected_ensemble_action="Temporary crisis, shift to non-USDC stablecoins",
    ),
    CrisisEvent(
        name="SEC Lawsuits",
        start="2023-06-01", end="2023-07-31",
        crisis_type="regulatory", severity=3,
        btc_drawdown_pct=-8.0,
        description="SEC sues Binance and Coinbase. Regulatory uncertainty spike.",
        expected_hmm_regime="normal to crisis transition",
        expected_ensemble_action="Moderate de-risking, not full defensive",
    ),
    CrisisEvent(
        name="Curve Exploit",
        start="2023-07-30", end="2023-08-15",
        crisis_type="protocol_exploit", severity=3,
        btc_drawdown_pct=-5.0,
        description="Curve Finance Vyper compiler exploit. $70M at risk. DeFi contagion fears.",
        expected_hmm_regime="normal (localised event)",
        expected_ensemble_action="Reduce stETH/rETH weight (DeFi exposure), minimal BTC impact",
    ),
    CrisisEvent(
        name="Israel-Hamas Conflict",
        start="2023-10-07", end="2023-10-31",
        crisis_type="macro", severity=2,
        btc_drawdown_pct=-5.0,
        description="Geopolitical shock. Brief risk-off across all assets.",
        expected_hmm_regime="normal (brief vol spike)",
        expected_ensemble_action="Temporary vol spike, quick recovery. Minimal rebalancing.",
    ),
    CrisisEvent(
        name="BTC ETF Launch Selloff",
        start="2024-01-10", end="2024-01-25",
        crisis_type="market_crash", severity=2,
        btc_drawdown_pct=-20.0,
        description="'Sell the news' after spot BTC ETF approval. BTC $49K to $39K.",
        expected_hmm_regime="normal to bear transition",
        expected_ensemble_action="Moderate de-risking if HMM detects transition",
    ),
    CrisisEvent(
        name="Yen Carry Unwind",
        start="2024-08-05", end="2024-08-20",
        crisis_type="macro", severity=3,
        btc_drawdown_pct=-18.0,
        description="BOJ rate hike triggers global carry trade unwind. Flash crash across assets.",
        expected_hmm_regime="crisis (brief but severe)",
        expected_ensemble_action="Circuit breaker may trigger on intraday drawdown",
    ),
]


def get_crisis_by_severity(severity_label: str) -> List[CrisisEvent]:
    """
    Filter crises by severity classification.

    Args:
        severity_label: One of 'mild', 'moderate', 'severe'

    Returns:
        List of CrisisEvent matching the severity label
    """
    return [c for c in CRYPTO_CRISES if c.severity_label == severity_label]


def get_crisis_mask(dates, crisis: Optional[CrisisEvent] = None) -> pd.Series:
    """
    Return boolean Series for timestamps falling within crisis period(s).

    If crisis is None, returns a mask covering ALL defined crises.

    Args:
        dates: DatetimeIndex or list of timestamps
        crisis: Optional single CrisisEvent. If None, uses all crises.

    Returns:
        pd.Series of bool, True where the timestamp falls in a crisis window
    """
    dates_idx = pd.DatetimeIndex(dates)
    mask = pd.Series(False, index=dates_idx)

    if crisis is not None:
        crises = [crisis]
    else:
        crises = CRYPTO_CRISES

    for c in crises:
        start = pd.Timestamp(c.start)
        end = pd.Timestamp(c.end)
        # Handle timezone-aware indices
        if dates_idx.tz is not None:
            start = start.tz_localize(dates_idx.tz)
            end = end.tz_localize(dates_idx.tz)
        mask = mask | ((dates_idx >= start) & (dates_idx <= end))

    return mask


def get_regime_periods(dates) -> dict:
    """Classify all timestamps into crisis / non-crisis periods."""
    mask = get_crisis_mask(dates)

    return {
        "crisis_indices": list(np.where(mask.values)[0]),
        "non_crisis_indices": list(np.where(~mask.values)[0]),
        "n_crises": len(CRYPTO_CRISES),
        "crisis_hours": int(mask.sum()),
        "total_hours": len(dates),
        "crisis_pct": float(mask.mean()) if len(dates) > 0 else 0,
    }


def crisis_stratified_metrics(returns, dates, metric_func) -> dict:
    """
    Compute any metric separately for crisis and non-crisis periods.

    Args:
        returns: Array of portfolio returns
        dates: Array of timestamps
        metric_func: Function that takes returns array and returns a metric value

    Returns:
        Dict with 'all', 'crisis', 'non_crisis' metric values
    """
    periods = get_regime_periods(dates)

    crisis_rets = returns[periods["crisis_indices"]] if periods["crisis_indices"] else np.array([0])
    non_crisis_rets = returns[periods["non_crisis_indices"]] if periods["non_crisis_indices"] else np.array([0])

    return {
        "all": float(metric_func(returns)),
        "crisis": float(metric_func(crisis_rets)) if len(crisis_rets) > 10 else None,
        "non_crisis": float(metric_func(non_crisis_rets)) if len(non_crisis_rets) > 10 else None,
        "n_crisis_hours": len(periods["crisis_indices"]),
        "n_non_crisis_hours": len(periods["non_crisis_indices"]),
    }


def get_crisis_summary() -> pd.DataFrame:
    """Return a summary DataFrame of all crisis events with severity labels."""
    rows = []
    for c in CRYPTO_CRISES:
        rows.append({
            "name": c.name,
            "start": c.start,
            "end": c.end,
            "type": c.crisis_type,
            "severity_score": c.severity,
            "severity_label": c.severity_label,
            "btc_drawdown_pct": c.btc_drawdown_pct,
            "duration_days": c.duration_days,
        })
    return pd.DataFrame(rows)
