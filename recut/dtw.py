from __future__ import annotations

import math
from typing import Callable, Iterable, List, Sequence, Tuple

import numpy as np


def cdist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute pairwise distances (cosine) between two sets of vectors.
    a: (N, D), b: (M, D) -> (N, M)
    """
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("a and b must be 2D arrays")
    # Normalize for cosine distance
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    sim = a_n @ b_n.T
    dist = 1.0 - sim
    return np.clip(dist, 0.0, 2.0)


def dtw_distance(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    window: int | None = None,
) -> Tuple[float, np.ndarray]:
    """Compute DTW distance with optional Sakoe-Chiba band.
    Returns (cost, path) where path is array of (i,j) indices.
    """
    N, M = len(seq_a), len(seq_b)
    D = cdist(seq_a, seq_b)
    if window is None:
        window = max(N, M)
    window = max(window, abs(N - M))

    C = np.full((N + 1, M + 1), np.inf, dtype=np.float32)
    C[0, 0] = 0.0
    for i in range(1, N + 1):
        j_start = max(1, i - window)
        j_end = min(M, i + window)
        for j in range(j_start, j_end + 1):
            c = D[i - 1, j - 1]
            C[i, j] = c + min(C[i - 1, j], C[i, j - 1], C[i - 1, j - 1])

    cost = float(C[N, M]) / (N + M)

    # backtrack path
    i, j = N, M
    path = []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        steps = [C[i - 1, j], C[i, j - 1], C[i - 1, j - 1]]
        argmin = int(np.argmin(steps))
        if argmin == 0:
            i -= 1
        elif argmin == 1:
            j -= 1
        else:
            i -= 1
            j -= 1
    path.reverse()
    return cost, np.array(path, dtype=np.int32)


def best_alignment_offset(
    ref_seq: np.ndarray,
    src_seq: np.ndarray,
    search_from: int,
    search_to: int,
    window: int | None = None,
) -> Tuple[int, float]:
    """Slide ref_seq over src_seq[search_from:search_to] to find best DTW cost.
    Returns (best_offset_start, best_cost)
    """
    best_cost = float("inf")
    best_off = search_from
    for off in range(search_from, max(search_from + 1, search_to - len(ref_seq) + 1)):
        seg = src_seq[off : off + len(ref_seq)]
        if len(seg) < max(2, len(ref_seq) // 3):
            break
        cost, _ = dtw_distance(ref_seq, seg, window=window)
        if cost < best_cost:
            best_cost = cost
            best_off = off
    return best_off, best_cost

