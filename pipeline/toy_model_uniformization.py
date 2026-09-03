import argparse
import uniformization
import os
import time as _time
import pandas as pd
import numpy as np


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1), help="number of worker processes (default: all but one core)")
    parser.add_argument("--Ns", type=int, nargs="+", default=[5, 50, 500, 5000], help="population sizes to compute")
    args = parser.parse_args()

    from joblib import Parallel, delayed

    print(f"[setup] detected {os.cpu_count()} cores, using n_jobs={args.jobs}", flush=True)

    for N in args.Ns:
        eps = 1e-12  # target total L1 error from Poisson truncation
        M = 2 * N   #Chosen truncation
        times = np.linspace(0, 2.5 * np.log(N), 1000)

        print(f"[N={N}] M={M}, {len(times)} time points -> launching workers...", flush=True)
        t0 = _time.time()

        expect = Parallel(n_jobs=args.jobs, verbose=10)(delayed(uniformization._e_xt_at)(N, M, t, eps) for t in times)

        dt = _time.time() - t0
        df = pd.DataFrame({'Number of Particles': N, 'Bound': M, 'Error': eps, 'Time': times, 'Numerical': expect})
        out = f'Numerical_x_N{N}.csv'
        df.to_csv(out)
        print(f"[N={N}] done in {dt:.1f}s -> wrote {out}", flush=True)

    print("Finished !", flush=True)
