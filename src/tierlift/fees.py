"""Fee schedule logic: tiers, blended rates, and distance to the next tier.

Every function is vectorised (NumPy in, NumPy out), because the simulator and
the analyses call them on thousands of clients at once.

Vocabulary
----------
* **bps**: basis point, 1/100 of a percent. A 10 bps fee on $1M is $1,000.
* **maker / taker**: a maker order rests on the book and adds liquidity, so it
  pays the lower fee. A taker order removes liquidity and pays the higher fee.
* **blended rate**: maker_ratio * maker_fee + (1 - maker_ratio) * taker_fee.
* **distance to next tier**: (next_threshold - volume) / next_threshold, so
  0.15 means "15% short of the next tier's volume threshold".
"""

from __future__ import annotations

import numpy as np

from tierlift.config import Config, Tier

BPS = 1e-4

DISTANCE_BUCKETS = ("within_10", "10_to_20", "20_plus", "top_tier")


class FeeSchedule:
    """Vectorised view of the tier table in the config."""

    def __init__(self, tiers: tuple[Tier, ...]):
        self.tiers = tiers
        self.names = np.array([t.name for t in tiers])
        self.thresholds = np.array([t.min_volume for t in tiers], dtype=float)
        self.maker = np.array([t.maker_bps for t in tiers]) * BPS
        self.taker = np.array([t.taker_bps for t in tiers]) * BPS
        self.top = len(tiers) - 1

    @classmethod
    def from_config(cls, cfg: Config) -> FeeSchedule:
        return cls(cfg.tiers)

    def tier_index(self, quarterly_volume: np.ndarray) -> np.ndarray:
        """0 = lowest tier. A volume exactly on a threshold earns the higher tier."""
        v = np.asarray(quarterly_volume, dtype=float)
        return np.searchsorted(self.thresholds, v, side="right") - 1

    def blended_rate(self, tier_idx: np.ndarray, maker_ratio: np.ndarray) -> np.ndarray:
        t = np.asarray(tier_idx)
        m = np.asarray(maker_ratio, dtype=float)
        return m * self.maker[t] + (1 - m) * self.taker[t]

    def next_threshold(self, tier_idx: np.ndarray) -> np.ndarray:
        """Volume needed for the next tier; NaN for the top tier."""
        t = np.asarray(tier_idx)
        out = np.full(t.shape, np.nan)
        below_top = t < self.top
        out[below_top] = self.thresholds[t[below_top] + 1]
        return out

    def distance_to_next(self, quarterly_volume: np.ndarray) -> np.ndarray:
        """Fractional shortfall to the next tier; NaN for the top tier."""
        v = np.asarray(quarterly_volume, dtype=float)
        nxt = self.next_threshold(self.tier_index(v))
        return (nxt - v) / nxt


def distance_bucket(distance_pct: np.ndarray) -> np.ndarray:
    """Map the fractional distance to the next tier into the four analysis buckets."""
    d = np.asarray(distance_pct, dtype=float)
    return np.select(
        [np.isnan(d), d <= 0.10, d <= 0.20],
        ["top_tier", "within_10", "10_to_20"],
        default="20_plus",
    )
