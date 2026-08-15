import wntr
from typing import List, Dict, Sequence
import numpy as np

def _pattern_len(wn):
    dur   = wn.options.time.duration
    pstep = wn.options.time.pattern_timestep
    return int(dur // pstep) + 1

def _fit_to_len(seq, n):
    if seq is None: return [1.0]*n
    seq = list(seq)
    if len(seq) < n:  seq = seq + [seq[-1]]*(n-len(seq))
    if len(seq) > n:  seq = seq[:n]
    return seq

def make_bounds_for_curve_choices(items: List[str], candidates: Sequence[str] | Dict[str, Sequence[str]]):
    """
    Return a bounds list aligned with items (each as an integer interval (0, K_i - 1))
    """
    bounds = []
    shared = isinstance(candidates, (list, tuple))
    for nm in items:
        k = len(candidates) if shared else len(candidates.get(nm, []))
        if k == 0:
            raise ValueError(f"No candidate curve name provided for tank {nm}")
        bounds.append((0, k - 1))
    return bounds

# --- selection context utils ---
def _opt_ctx(wn: wntr.network.WaterNetworkModel) -> dict:
    if not hasattr(wn, "_opt_ctx"):
        wn._opt_ctx = {}
    return wn._opt_ctx


def _mean_or_inf(x):
    try:
        return float(np.nanmean(np.asarray(x).astype(float)))
    except Exception:
        return np.inf

