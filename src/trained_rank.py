"""K.4 --- the rank of the **TRAINED** dW.  Anti-cheating test 2b, as written.

    env/bin/python src/trained_rank.py --dir scratchpad/k4_theta

WHY THIS FILE EXISTS
--------------------
Anti-cheating test 2b (`llmdocs/PROMPT.md`) asks for the effective rank of the
**trained** dW at matched `k`, against the FourierFT arm.  **Every rank number
this program has published is measured at INITIALISATION**, because no adapter
checkpoint was ever saved.  Both arms were measured identically, so past
*comparisons* are fair -- but the test as written had never been run.

Both arms are exactly determined by a fixed support plus `k` real scalars:

    bWHT      dW = A_m^T C(theta) A_n,  A = (I (x) H_B) P  orthogonal, support fixed
    FourierFT dW = ifft2(S(theta)).real * scaling,          indices fixed by seed 777

so 1000 floats per module is a COMPLETE checkpoint for every rank statistic.
`train_glue.py --save_adapter_dir` (K.4, default OFF) dumps exactly that at init
and at the end of every epoch; this file turns those dumps into the statistics.

WHAT IT REPORTS, per module, per snapshot
-----------------------------------------
  * stable rank  ||dW||_F^2 / ||dW||_2^2          )  all three from
  * erank = exp(spectral entropy)                 )  `src/effective_rank.rank_stats`
  * numerical rank (LAPACK convention)            )  -- the program's instrument
  * PR/d^2 of dW = (sum dW^2)^2 / (d^2 sum dW^4)  -- verify_k1_grid.py's `pr_norm`
  * PR_theta/k   = (sum th^2)^2 / (k sum th^4)    -- the VALUE-DISTRIBUTION stat
        == 1.0 for Rademacher (equal magnitude), ~1/3 for Gaussian.

BUILT-IN CONSISTENCY CHECKS (printed, not assumed)
--------------------------------------------------
  C1  the support tables in the dump are reproduced bit-exactly by re-drawing
      them from the recorded seed with the SHIPPED `bwht_adapter` / PEFT code
      => the dump really is the object that trained.
  C2  sv(dW) == sv(C) for the bWHT arm (A orthogonal) -- a live orthogonality
      check on the reconstruction, not an assumption.
  C3  the a-priori atom norm  scaling*sqrt(mu)  reproduces 0.138106793200498.

NOTHING here is trained, swept or tuned; it is a read-only measurement.
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bwht_adapter import SUPPORTS, bwht_matrix
from effective_rank import peft_indices, rank_stats

ATOM_TARGET = 0.138106793200498


# --------------------------------------------------------------------------- #
#  statistics                                                                  #
# --------------------------------------------------------------------------- #

def pr_norm(dW: torch.Tensor) -> float:
    """(sum dW^2)^2 / (numel * sum dW^4)  --  verify_k1_grid.py's `pr_norm`."""
    a = dW.to(torch.float64)
    s2 = float((a ** 2).sum())
    s4 = float((a ** 4).sum())
    return s2 * s2 / (a.numel() * s4)


def pr_theta(theta: torch.Tensor) -> float:
    """(sum th^2)^2 / (k * sum th^4).  1.0 = Rademacher, ~1/3 = Gaussian."""
    return pr_norm(theta.reshape(-1))


# --------------------------------------------------------------------------- #
#  dW reconstruction (exact, from theta + the fixed support)                   #
# --------------------------------------------------------------------------- #

_A_CACHE: Dict[Tuple[int, int, int], torch.Tensor] = {}


def _A(d: int, B: int, seed: int) -> torch.Tensor:
    key = (d, B, seed)
    if key not in _A_CACHE:
        _A_CACHE[key] = bwht_matrix(d, B, seed, torch.float64)
    return _A_CACHE[key]


def bwht_dW(meta: dict, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """dW = A_m^T C A_n and the core C, both fp64.  Mirrors
    `BwhtLinear.get_delta_weight` exactly (same expression, same order)."""
    m, n = meta["m"], meta["n"]
    C = torch.zeros(m, n, dtype=torch.float64)
    C = C.index_put((meta["rows"], meta["cols"]),
                    (theta.to(torch.float64) * meta["scaling"])[meta["pidx"]])
    Am, An = _A(m, meta["block"], meta["support_seed"]), _A(n, meta["block"], meta["support_seed"])
    return Am.T @ C @ An, C


def fourierft_dW(meta: dict, theta: torch.Tensor) -> torch.Tensor:
    """dW = ifft2(dense).real * scaling -- PEFT `get_delta_weight` verbatim."""
    m, n = meta["m"], meta["n"]
    idx = meta["indices"]
    dense = torch.zeros(m, n, dtype=torch.float64)
    dense[idx[0].long(), idx[1].long()] = theta.to(torch.float64)
    return torch.fft.ifft2(dense).real * meta["scaling"]


# --------------------------------------------------------------------------- #
#  consistency checks                                                          #
# --------------------------------------------------------------------------- #

def check_support(meta: dict) -> str:
    """C1: re-draw the support from the recorded seed with the shipped code."""
    if meta["kind"] == "bwht":
        r, c, p = SUPPORTS[meta["support"]](meta["m"], meta["n"], meta["k"],
                                            meta["mu"], meta["support_seed"])
        ok = (torch.equal(r.long(), meta["rows"].long())
              and torch.equal(c.long(), meta["cols"].long())
              and torch.equal(p.long(), meta["pidx"].long()))
        return "EXACT" if ok else "MISMATCH"
    idx = peft_indices(meta["m"], meta["n"], meta["k"], meta["random_loc_seed"])
    return "EXACT" if torch.equal(idx.long(), meta["indices"].long()) else "MISMATCH"


# --------------------------------------------------------------------------- #
#  the sweep                                                                   #
# --------------------------------------------------------------------------- #

def tag_order(tag: str) -> Tuple[int, int]:
    if tag == "init":
        return (0, -1)
    mm = re.fullmatch(r"ep(\d+)", tag)
    return (1, int(mm.group(1))) if mm else (2, 0)


def run_stem(d: str, stem: str, tags: Optional[List[str]] = None,
             modules: Optional[int] = None, verbose: bool = True) -> List[dict]:
    meta = torch.load(os.path.join(d, f"{stem}_meta.pt"), weights_only=False)
    names = sorted(meta.keys())
    if modules:
        names = names[:modules]
    kind = meta[names[0]]["kind"]

    avail = []
    for f in glob.glob(os.path.join(d, f"{stem}_*.pt")):
        t = os.path.basename(f)[len(stem) + 1:-3]
        if t != "meta":
            avail.append(t)
    avail = sorted(set(avail), key=tag_order)
    if tags:
        avail = [t for t in avail if t in tags]

    if verbose:
        m0 = meta[names[0]]
        print(f"\n{'=' * 100}\nARM: {stem}   kind={kind}   modules={len(names)}   "
              f"snapshots={len(avail)}")
        cfg = {k: v for k, v in m0.items()
               if k in ("m", "n", "k", "mu", "block", "support_seed", "support",
                        "scaling", "random_loc_seed")}
        print(f"  config (module 0): {cfg}")
        # C1
        st = {check_support(meta[nm]) for nm in names}
        print(f"  C1 support re-draw from the recorded seed, all {len(names)} modules: "
              f"{'/'.join(sorted(st))}")
        # C3
        if kind == "bwht":
            atom = m0["scaling"] * math.sqrt(m0["mu"])
            print(f"  C3 a-priori atom norm scaling*sqrt(mu) = {atom:.15f} "
                  f"(target {ATOM_TARGET:.15f}, rel {abs(atom - ATOM_TARGET) / ATOM_TARGET:.2e})")
        # supports identical across modules?
        key = "rows" if kind == "bwht" else "indices"
        same = all(torch.equal(meta[nm][key], meta[names[0]][key]) for nm in names)
        print(f"  support identical across all modules: {same}")
        print(f"{'=' * 100}")

    th0 = torch.load(os.path.join(d, f"{stem}_init.pt"), weights_only=False)

    rows_out: List[dict] = []
    sv_check = 0.0
    for tag in avail:
        th = torch.load(os.path.join(d, f"{stem}_{tag}.pt"), weights_only=False)
        for nm in names:
            mt, theta = meta[nm], th[nm].to(torch.float64)
            t0 = th0[nm].to(torch.float64)
            # The COUNTERFACTUAL: the same signs with every magnitude equalised
            # (a Rademacher value distribution at this point of training).  It
            # measures how much stable rank the realised magnitude SPREAD costs.
            # It is a measurement of a well-defined object, NOT an achievable
            # configuration -- see the caveat in llmdocs/K4_trained_rank.md.
            th_eq = torch.sign(theta)
            if kind == "bwht":
                dW, C = bwht_dW(mt, theta)
                sv = torch.linalg.svdvals(dW)
                svC = torch.linalg.svdvals(C)
                sv_check = max(sv_check,
                               float((sv - svC).abs().max() / svC.max()))
                sv_eq = torch.linalg.svdvals(bwht_dW(mt, th_eq)[0])
            else:
                dW = fourierft_dW(mt, theta)
                sv = torch.linalg.svdvals(dW)
                sv_eq = torch.linalg.svdvals(fourierft_dW(mt, th_eq))
            rs = rank_stats(dW, sv=sv)
            eq = float((sv_eq ** 2).sum() / sv_eq.max() ** 2)
            # --- memory of the INIT: is the end state set by theta_0 or by AdamW? --
            n0, n1 = float(t0.norm()), float(theta.norm())
            cos = float((t0 @ theta) / (n0 * n1)) if n0 * n1 > 0 else float("nan")
            tc, thc = t0 - t0.mean(), theta - theta.mean()
            corr = float((tc @ thc) / (tc.norm() * thc.norm()))
            rows_out.append(dict(
                stem=stem, kind=kind, tag=tag, module=nm,
                mu=mt.get("mu", 1), k=mt["k"],
                # the VALUE-TAIL ceiling on stable rank (K.2 3.2):
                #   ||C||_2 >= max_ij|C_ij|  =>  stable rank <= mu*sum(th^2)/max(th^2)
                rank_ceiling=mt.get("mu", 1) * float((theta ** 2).sum())
                              / float((theta ** 2).max()),
                stable=rs["stable_rank"], erank=rs["erank"],
                numrank=rs["numerical_rank"], stable_eqmag=eq,
                pr_dW=pr_norm(dW), pr_theta=pr_theta(theta),
                theta_norm=n1, norm_ratio=n1 / n0,
                displacement=float((theta - t0).norm()) / n0,
                cos_init=cos, corr_init=corr,
                sign_agree=float((torch.sign(theta) == torch.sign(t0)).double().mean()),
                theta_absmean=float(theta.abs().mean()),
                theta_max=float(theta.abs().max()),
                theta_kurt=float(((theta / theta.std()) ** 4).mean()),
                fro=rs["fro"], smax=rs["sigma_max"],
            ))
        if verbose:
            print(f"    ... {tag} done", flush=True)
    if verbose and kind == "bwht":
        print(f"  C2 max rel |sv(dW) - sv(C)| over every module x snapshot: "
              f"{sv_check:.3e}   (A orthogonal => must be ~1e-15)")
    return rows_out


def summarise(rows: List[dict]) -> None:
    import collections
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["stem"], r["tag"])].append(r)
    stems = sorted({r["stem"] for r in rows})
    for stem in stems:
        tags = sorted({t for s, t in by if s == stem}, key=tag_order)
        print(f"\n{'-' * 118}\n{stem}: mean +- sd over modules\n{'-' * 118}")
        hdr = (f"{'snapshot':>9} {'stable rank':>19} {'erank':>19} {'numrank':>19} "
               f"{'PR/d^2 of dW':>17} {'PR_theta/k':>17} {'||th||/||th0||':>16} "
               f"{'corr(th,th0)':>15} {'sign agree':>12} {'stable|eqmag':>13}")
        print(hdr)
        base = None
        for t in tags:
            v = by[(stem, t)]
            def ms(kk, f="{:.2f}"):
                a = np.array([x[kk] for x in v], dtype=float)
                return (f + " +- " + f).format(a.mean(), a.std(ddof=1))
            def mo(kk, f="{:.3f}"):
                return f.format(np.array([x[kk] for x in v], dtype=float).mean())
            print(f"{t:>9} {ms('stable'):>19} {ms('erank'):>19} {ms('numrank'):>19} "
                  f"{ms('pr_dW', '{:.5f}'):>17} {ms('pr_theta', '{:.4f}'):>17} "
                  f"{mo('norm_ratio'):>16} {mo('corr_init'):>15} {mo('sign_agree'):>12} "
                  f"{mo('stable_eqmag', '{:.2f}'):>13}")
            if t == "init":
                base = v
        if base is not None and len(tags) > 1:
            last = by[(stem, tags[-1])]
            def rat(kk):
                a = np.array([x[kk] for x in last]).mean()
                b = np.array([x[kk] for x in base]).mean()
                return a / b if b else float("nan")
            print(f"{'ratio':>9} {rat('stable'):>19.4f} {rat('erank'):>19.4f} "
                  f"{rat('numrank'):>19.4f} {rat('pr_dW'):>17.4f} "
                  f"{rat('pr_theta'):>17.4f}   <- {tags[-1]} / init")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--stems", nargs="*", default=None)
    ap.add_argument("--tags", nargs="*", default=None)
    ap.add_argument("--modules", type=int, default=None,
                    help="restrict to the first N modules (smoke tests only)")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    stems = args.stems or sorted(
        os.path.basename(f)[:-8] for f in glob.glob(os.path.join(args.dir, "*_meta.pt")))
    rows: List[dict] = []
    for s in stems:
        rows += run_stem(args.dir, s, args.tags, args.modules)
    summarise(rows)
    if args.csv:
        import pandas as pd
        pd.DataFrame(rows).to_csv(args.csv, index=False)
        print(f"\nper-module rows -> {args.csv}")


if __name__ == "__main__":
    main()
