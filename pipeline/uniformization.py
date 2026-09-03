"""
This script carries out Uniformization for a given birth_death process, parallelises over several CPUs for efficiency and uses Numba Jit accelerator.
"""

import argparse
import math
import os
import time as _time
from typing import Tuple, Dict

from numba import jit
import numpy as np
import pandas as pd

@jit(nopython=True)
def build_rates(N: float, M: int) -> Tuple[np.ndarray, np.ndarray]: # Generator on truncated state space {1,...,M}
    """A function that initialises the birth rate, mu, and death rate, lam, as ndarrays."""
    lam, mu = np.zeros(M+2, dtype=np.float64), np.zeros(M+2, dtype=np.float64)
    lam[:M+1] = np.arange(M+1, dtype=np.float64)
    mu[:M+1]  = (np.arange(M+1, dtype=np.float64) * (np.arange(M+1, dtype=np.float64) - 1.0)) / float(N)
    lam[M+1] = 0.0
    mu[M+1]  = 0.0

    return lam, mu #Splitting, coalescence rates resp.

@jit(nopython=True)
def choose_B(lam: np.ndarray, mu: np.ndarray) -> float:
    """A function that returns the uniformising parameter, taken as at least the maximal rate."""
    return float(np.max(lam + mu))      # Unformizing scalar B

@jit(nopython=True)
def row_tridiag_multiply_P(v: np.ndarray, lam: np.ndarray, mu: np.ndarray, B: float) -> np.ndarray:
    """The matrix multiplication of some vector v, with the infinitesimal generator of this birth death process."""
    out = v * (1.0 - (lam + mu) / B)            # P = I + Q/B, we compute out = v * P
    out[1:]  += v[:-1] * (lam[:-1] / B)
    out[:-1] += v[1:]  * (mu[1:]  / B)
    return out

@jit(nopython=True)
def uniformize_step(v0: np.ndarray, lam: np.ndarray, mu: np.ndarray, B: float, dt: float, eps: float):
    """Computing the sum of terms associated to this poisson process, up to eps, effectively uniformising the CTMC."""
    lam_pois = B * dt  # dt a time step
    if lam_pois == 0.0:
        return v0.copy(), 0, 0.0
    w = np.exp(-lam_pois)
    v = v0.copy()
    accum = w * v
    comp = np.zeros_like(accum)   # Kahan compensation for accum
    cdf = w                   # cumulative Poisson mass up to k=0
    K = 0
    while (1.0 - cdf) > eps:    # eps bounds truncated tail mass
        K += 1
        v = row_tridiag_multiply_P(v, lam, mu, B)
        w = w * lam_pois / K
        y = w * v - comp
        s = accum + y
        comp = (s - accum) - y
        accum = s
        cdf += w
        if K > 2_000_000:     # safety
            break
    tail = max(0.0, 1.0 - cdf)
    return accum, K, tail

def uniformize_step_wrapped(*args):
    accum, K, tail = uniformize_step(*args)
    return accum, {"K": K, "tail": tail}   # dict assembled outside nopython

@jit(nopython=True)
def uniformize_to_time(v0: np.ndarray, lam: np.ndarray, mu: np.ndarray, B: float, t: float, eps: float, lam_max: float):
    """Splice time to ensure numerical stability of each sum."""
    if t == 0.0:    # As explained in report, we splice time to ensure numerical stability of each sum computed.
        return v0.copy(), [0], [0.0], 1
    num_slices = int(np.ceil((B * t) / lam_max)) if B * t > lam_max else 1
    dt = t / num_slices
    eps_slice = max(eps / num_slices, 1e-14)
    v = v0.copy()
    K_list, tail_list = [], []
    for _ in range(num_slices):
        v, K, tail = uniformize_step(v, lam, mu, B, dt, eps_slice)
        K_list.append(K)
        tail_list.append(tail)
    return v, K_list, tail_list, num_slices

def uniformize_to_time_wrapped(*args):
    v, K_list, tail_list, num_slices = uniformize_to_time(*args)
    return v, {"K_list": K_list, "tail_list": tail_list, "slices": num_slices}

def initial_distribution(state_size: int, i0: int = 1) -> np.ndarray:
    v = np.zeros(state_size, dtype=float)
    i0 = min(max(0, i0), state_size - 1)
    v[i0] = 1.0
    return v

def expect_x(p: np.ndarray) -> float:
    i = np.arange(p.size, dtype=float)
    return float(math.fsum(i * p))      # Expected value

def solve_bd_ctmc(N: int, M: int, t: float, eps: float = 1e-12, overflow: bool = True, i0: int = 1, lam_max: float = 300.0, B: float = None) -> Dict:
    lam, mu = build_rates(N, M)
    if B is None:
        B = choose_B(lam, mu)
    p0 = initial_distribution(len(lam), i0=i0)
    p_t, info = uniformize_to_time_wrapped(p0, lam, mu, B, t, eps, lam_max)
    return {
        "p_t": p_t,
        "E_Xt": expect_x(p_t),
        "overflow_mass": float(p_t[-1]) if overflow else 0.0,
        "B": B,
        "info": info,
        "state_count": len(lam),
        "M": M
    }

def _e_xt_at(N: int, M: int, t: float, eps: float) -> float:
    # Picklable top-level worker: one independent fresh propagation to time t.
    return solve_bd_ctmc(N, M, t, eps)["E_Xt"]


