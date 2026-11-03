# Vessel imaging analysis methods.
from __future__ import annotations
import numpy as np
import pandas as pd

def holm_adjust(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full(p.shape, np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    if not valid.size:
        return adjusted.tolist()
    order = valid[np.argsort(p[valid])]
    running = 0.0
    m = len(order)
    for rank, index in enumerate(order):
        value = min(1.0, (m - rank) * p[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()

def add_holm(results: pd.DataFrame) -> pd.DataFrame:
    out = results.copy()
    out["Holm-adjusted p"] = holm_adjust(out["Raw p"].tolist())
    out["Significance"] = out["Holm-adjusted p"].map(significance_label)
    return out

def significance_label(p: float) -> str:
    if not np.isfinite(p):
        return "not tested"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"
