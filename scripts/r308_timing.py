#!/usr/bin/env python
"""[R.308] MODULE-LEVEL WALL-CLOCK AND ADAPTER MEMORY, all nine arms.

  env/bin/python scripts/r308_timing.py --selftest      # before believing it
  env/bin/python scripts/r308_timing.py --run           # measure  (writes CSV)
  env/bin/python scripts/r308_timing.py                 # read the CSV

WHY THIS EXISTS
  `[R.101]`'s "4.8% slower" was WITHDRAWN (two forced GPU syncs, since fixed) and
  CONTEXT 5 bars any replacement "until [R.103] lands".  This is that measurement.
  Without it the joint table has a hole exactly where the interesting metric is:
  every cost column in `[R.307]` is a DERIVED op-count, and `[P.2]` established
  that this family is DISPATCH-BOUND -- flops and wall-clock give DIFFERENT orders.

  `src/bench_adapter_cost.py` is the J.2 rig and its hygiene rules are inherited
  verbatim (below).  It could not be used directly: its executable arm list builds
  only 5 modules and none of LoCA / QWHA / WaveFT / LYRA / SCoRA.

MEASUREMENT HYGIENE (inherited from `src/bench_adapter_cost.py`, J.2)
  * CUDA events + explicit synchronize around every timed region
  * WARMUP iterations discarded per (arm, tf32, mode); median over REPS
  * the WHOLE sweep repeated REPEATS times, arm order RANDOMISED per block, and
    the reported estimator is the MINIMUM block-median across repeats -- on shared
    hardware interference is strictly additive, so the minimum is the defensible
    estimator.  The spread across repeats is recorded so contention is visible.
  * identical dtype / shape / device / input tensor object across arms
  * BOTH TF32 settings swept: whether the dense GEMM gets tensor cores changes the
    answer, and hiding that would be dishonest.

⛔ THE GATE THIS RUN CANNOT SKIP
  Every arm is checked for hidden device syncs AT MEASUREMENT TIME
  (`torch.cuda.set_sync_debug_mode("warn")`, 0 syncs required per forward).
  `[R.101]` and `[R.104]` were the same defect written twice by different authors;
  `src/verify_no_sync.py` is the standing static gate and `[R.308]` extended it to
  the BASELINES (loca/qwha/haar), because a hidden sync in a BASELINE makes that
  baseline look slow and flatters us -- the mirror image of the original defect,
  and precisely CONTEXT 5's "never quote a ratio off an asymmetrically-repaired
  pair".  An arm that fails the sync check is recorded as FAILED, not timed.

⛔ WHAT A NUMBER FROM THIS FILE IS AND IS NOT
  It is the MARGINAL cost of one adapted module over a frozen `nn.Linear` of the
  same shape.  It is NOT an end-to-end training speedup: the adapter is 10-13% of
  training wall-clock (`[P.5-P.11]`), so any module-level ratio here is capped at
  1.11-1.16x end-to-end.  The reader prints that conversion rather than leaving it
  to the reader of the reader.
"""
import argparse, csv, gc, math, os, random, statistics, sys, warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

OUT = os.path.join(ROOT, "scratchpad", "phaseR", "r308", "timing.csv")

D_MODEL   = 768
K         = 256
B_TOKENS  = 4096          # 32 x 128, the batch [R.305] measured accuracy at
WARMUP    = 20
REPS      = 50
REPEATS   = 5             # whole-sweep repeats; estimator is the MIN block-median

# arm key -> (title, builder).  Every builder takes a frozen nn.Linear and returns
# a module whose forward is EXACTLY the path [R.305]/[R.306] ran.
# `scora` and `scora2` are the SAME forward path (they differ only in `lr` and
# `scaling`, neither of which changes an op count or a kernel launch), so the two
# accuracy rows share ONE timing measurement rather than pretending to differ.
ALIAS = {"scora2": "scora",
         # `fftstock` is not timed (see ARM_ORDER): it is represented by the two
         # sync-free implementations of the same math this repo owns.
         "fftstock": "fftfast"}

# ⛔ `fftstock` is PEFT's own `FourierFTLinear` and carries 1 device sync per
# forward.  It is NOT repaired here (it is third-party, installed, and [R.95]
# gates this repo's FourierFT numbers as bit-identical to it), so it is NOT
# timed.  FourierFT is represented instead by TWO sync-free implementations of
# the same math that this repo owns -- `fftm` (merged, the row [R.305] reports)
# and `fftfast` -- so the method is not disadvantaged by its vendored code.
ARM_ORDER = ["frozen", "fftm", "fftfast", "loca", "qwha",
             "scora", "scora_factored", "wave1", "wave2", "lyra"]

TITLES = {
    "frozen":         "frozen nn.Linear (reference)",
    "fftm":           "FourierFT (merged)",
    "fftfast":        "FourierFT (fourierft-fast, rfft)",
    "loca":           "LoCA",
    "qwha":           "QWHA",
    "scora":          "SCoRA (--slr_materialise 1, AS RUN)",
    "scora_factored": "SCoRA (--slr_materialise 0, floor)",
    "wave1":          "WaveFT mu=1 (published)",
    "wave2":          "WaveFT mu=2 (repo fix)",
    "lyra":           "LYRA",
}


def build(arm, device, dtype):
    import torch, torch.nn as nn
    base = nn.Linear(D_MODEL, D_MODEL, bias=True).to(device=device, dtype=dtype)
    for p in base.parameters():
        p.requires_grad_(False)
    if arm == "frozen":
        return base
    if arm == "fftm":
        from merged_fourierft import MergedFourierFTLinear
        return MergedFourierFTLinear(base, n_frequency=K, scaling=100.0,
                                     random_loc_seed=777).to(device)
    if arm == "fftstock":
        import peft.tuners.fourierft.layer as L
        m = L.FourierFTLinear(base, "default", n_frequency=K, scaling=100.0,
                              random_loc_seed=777, init_weights=False)
        return m.to(device)
    if arm == "loca":
        from loca_adapter import LoCALinear
        return LoCALinear(base, n_frequency=K, scale=0.25,
                          learn_location_iter=-1, init_seed=777).to(device)
    if arm == "qwha":
        from qwha_adapter import QWHALinear
        # ⛔ sync_free=True: the vendored forward's `sparse_coo_tensor` costs 2
        # device syncs (gate G8 asserts the two paths are bit-identical in forward
        # AND gradient).  Timing the syncing path would flatter US.
        return QWHALinear(base, n_frequency=K, scaling=26.5165,
                          random_loc_seed=777, init_weights=False,
                          sync_free=True).to(device)
    if arm in ("scora", "scora_factored"):
        from slr_adapter import SLRLinear
        return SLRLinear(base, rank=1, s=128, seed=777, init="zero",
                         materialise=(arm == "scora")).to(device)
    if arm in ("wave1", "wave2"):
        from haar_adapter import HaarLinear
        return HaarLinear(base, n_frequency=K, mu=1 if arm == "wave1" else 2,
                                   support_seed=777, fourierft_scaling=150.0,
                                   init_std=0.0).to(device)
    if arm == "fftfast":
        from fourierft_fast import FourierFTFastLinear
        return FourierFTFastLinear(base, n_frequency=K, scaling=100.0,
                                   random_loc_seed=777).to(device)
    if arm == "lyra":
        from spectral_adapter import SpectralAdapterLinear
        return SpectralAdapterLinear(base, p=16, q=16, scaling=0.05, d_initial=0.07,
                              freq_mode="geometric", freq_exponent=2.0).to(device)
    raise KeyError(arm)


# ============================================================================
# THE SYNC GATE -- run per arm, at measurement time
# ============================================================================
def sync_count(mod, x):
    """Device syncs during ONE forward, caches warmed.  0 is the only pass."""
    import torch
    for _ in range(3):                      # warm every per-instance cache first
        mod(x)
    torch.cuda.synchronize()
    n = 0
    torch.cuda.set_sync_debug_mode("warn")
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mod(x)
            n = sum(1 for r in w if "sync" in str(r.message).lower())
    finally:
        torch.cuda.set_sync_debug_mode("default")
    return n


# ============================================================================
# TIMING
# ============================================================================
def _time_block(mod, x, mode, reps, warmup):
    """Median ms over `reps` timed iterations of `mode` in {'fwd','fwdbwd'}."""
    import torch
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
          for _ in range(reps)]

    def one():
        if mode == "fwd":
            with torch.no_grad():
                mod(x)
        else:
            xr = x.detach().requires_grad_(True)
            out = mod(xr)
            out.square().mean().backward()
            mod.zero_grad(set_to_none=True)

    for _ in range(warmup):
        one()
    torch.cuda.synchronize()
    for a, b in ev:
        a.record(); one(); b.record()
    torch.cuda.synchronize()
    return statistics.median(a.elapsed_time(b) for a, b in ev)


def marginal_memory(mod, base_bytes, x):
    """Bytes still held by the allocator after the forward, MINUS the same for a
    frozen Linear of identical shape -- the adapter's true activation cost.
    [measured], never estimated (bench_adapter_cost's own definition)."""
    import torch
    gc.collect(); torch.cuda.empty_cache()
    xr = x.detach().requires_grad_(True)
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    out = mod(xr)
    held = torch.cuda.memory_allocated() - before
    del out
    gc.collect(); torch.cuda.empty_cache()
    return held - base_bytes


def run(device="cuda:0"):
    import torch
    torch.manual_seed(0)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rows = []
    dtype = torch.float32
    for tf32 in (False, True):
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
        x = torch.randn(B_TOKENS, D_MODEL, device=device, dtype=dtype)
        mods = {}
        for arm in ARM_ORDER:
            mods[arm] = build(arm, device, dtype)
        # ---- the gate, before any timing --------------------------------
        syncs = {}
        for arm, mod in mods.items():
            try:
                syncs[arm] = sync_count(mod, x)
            except Exception as e:            # never let the gate kill the run
                syncs[arm] = -1
                print(f"[r308] sync check ERROR {arm}: {e!r}")
        bad = [a for a, n in syncs.items() if n != 0]
        if bad:
            print(f"[r308] ⛔ SYNC GATE FAILED for {bad} -- those arms are NOT timed")
        # ---- marginal memory (once per tf32 block) -----------------------
        gc.collect(); torch.cuda.empty_cache()
        xr = x.detach().requires_grad_(True)
        b0 = torch.cuda.memory_allocated()
        o = mods["frozen"](xr); base_bytes = torch.cuda.memory_allocated() - b0
        del o, xr
        mem = {a: (marginal_memory(mods[a], base_bytes, x) if syncs.get(a) == 0 else None)
               for a in ARM_ORDER}
        # ---- timing: REPEATS blocks, arm order randomised each time ------
        for rep in range(REPEATS):
            order = [a for a in ARM_ORDER if syncs.get(a) == 0]
            random.Random(1000 + rep).shuffle(order)
            for arm in order:
                for mode in ("fwd", "fwdbwd"):
                    ms = _time_block(mods[arm], x, mode, REPS, WARMUP)
                    rows.append(dict(arm=arm, tf32=int(tf32), mode=mode, rep=rep,
                                     ms=ms, syncs=syncs[arm],
                                     mem_bytes=mem[arm] if mem[arm] is not None else ""))
            print(f"[r308] tf32={int(tf32)} rep {rep + 1}/{REPEATS} done")
        for arm in bad:
            rows.append(dict(arm=arm, tf32=int(tf32), mode="fwd", rep=-1, ms="",
                             syncs=syncs[arm], mem_bytes=""))
        del mods
        gc.collect(); torch.cuda.empty_cache()
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arm", "tf32", "mode", "rep", "ms",
                                          "syncs", "mem_bytes"])
        w.writeheader()
        w.writerows(rows)
    print(f"[r308] wrote {len(rows)} rows -> {OUT}")
    return rows


# ============================================================================
# READING -- the MIN block-median estimator, and the marginal over frozen
# ============================================================================
def load(path=None):
    path = path or OUT
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def estimate(rows, tf32=1):
    """{arm: {mode: (min_block_median_ms, spread_across_repeats)}}.

    ⭐ MIN across repeats, not mean: this GPU is shared and interference is
    strictly ADDITIVE, so the minimum is the only defensible estimator.  The
    spread is returned alongside so contention is visible rather than hidden."""
    out = {}
    for r in rows:
        if not r["ms"] or int(r["tf32"]) != tf32:
            continue
        out.setdefault(r["arm"], {}).setdefault(r["mode"], []).append(float(r["ms"]))
    return {a: {m: (min(v), max(v) - min(v)) for m, v in d.items()}
            for a, d in out.items()}


def marginal(rows, tf32=1):
    """{arm: {mode: ms above the frozen Linear}} -- the adapter's own cost."""
    est = estimate(rows, tf32)
    if "frozen" not in est:
        return {}
    return {a: {m: est[a][m][0] - est["frozen"][m][0] for m in est[a]}
            for a in est if a != "frozen"}


def report():
    rows = load()
    if not rows:
        print("[r308] no measurement yet -- run with --run")
        return
    print("=" * 96)
    print("[R.308] MEASURED module wall-clock and adapter memory")
    print(f"d={D_MODEL} k={K} b={B_TOKENS} float32 | {REPEATS} sweeps x {REPS} reps, "
          f"MIN block-median | marginal over a frozen nn.Linear")
    print("=" * 96)
    for tf32 in (1, 0):
        est, marg = estimate(rows, tf32), marginal(rows, tf32)
        if not est:
            continue
        fr = est.get("frozen", {})
        print(f"\n--- TF32 {'ON' if tf32 else 'OFF'} "
              f"(frozen Linear: fwd {fr.get('fwd', ('?',))[0]:.4f} ms, "
              f"fwd+bwd {fr.get('fwdbwd', ('?',))[0]:.4f} ms) ---")
        print(f"{'arm':38s} {'fwd ms':>9s} {'+bwd ms':>9s} {'marg fwd':>9s} "
              f"{'marg+bwd':>9s} {'xfrozen':>8s} {'spread':>7s}")
        order = sorted((a for a in est if a != "frozen"),
                       key=lambda a: marg[a].get("fwdbwd", math.inf))
        for a in order:
            e = est[a]
            xf = e["fwdbwd"][0] / fr["fwdbwd"][0] if fr else float("nan")
            print(f"{TITLES.get(a, a):38s} {e['fwd'][0]:9.4f} {e['fwdbwd'][0]:9.4f} "
                  f"{marg[a]['fwd']:9.4f} {marg[a]['fwdbwd']:9.4f} {xf:7.3f}x "
                  f"{e['fwdbwd'][1]:7.4f}")
    # ---- memory ----------------------------------------------------------
    mem = {}
    for r in rows:
        if r["mem_bytes"] and int(r["tf32"]) == 1:
            mem[r["arm"]] = int(float(r["mem_bytes"]))
    if mem:
        print("\nMARGINAL STASHED ACTIVATION MEMORY over a frozen Linear [measured]:")
        for a, v in sorted(mem.items(), key=lambda kv: kv[1]):
            if a == "frozen":
                continue
            print(f"   {TITLES.get(a, a):38s} {v/1024**2:9.3f} MiB")
    # ---- the conversion the reader must not have to do -------------------
    marg = marginal(rows, 1)
    if marg and "scora" in marg:
        pub = {a: v for a, v in marg.items()
               if not a.startswith("scora") and a != "frozen"}
        if pub:
            best = min(pub, key=lambda a: pub[a]["fwdbwd"])
            r_mod = pub[best]["fwdbwd"] / marg["scora"]["fwdbwd"]
            print(f"\nSCoRA (as run) vs the fastest published arm ({TITLES[best]}):")
            print(f"   module-level marginal ratio: {r_mod:.3f}x")
            print(f"   ⛔ END-TO-END that is capped at 1.11-1.16x [P.5-P.11]: the adapter")
            print(f"      is only 10-13% of training wall-clock, so a {r_mod:.2f}x module win")
            print(f"      is worth at most {1/(1 - 0.13*(1 - 1/max(r_mod, 1e-9))):.3f}x of a training run.")
    print("\n⛔ These are MODULE-LEVEL marginals, not end-to-end speedups, and they")
    print("   supersede nothing until read together with [P.2]'s dispatch-bound finding:")
    print("   ranking by flops and by wall-clock gives DIFFERENT orders in this family.")


# ============================================================================
def selftest():
    n = [0]

    def ck(c, m):
        n[0] += 1
        if not c:
            print("FAIL:", m); sys.exit(1)

    # -- 1. the cell must be the one the accuracy was measured at ------------
    ck(B_TOKENS == 4096 and D_MODEL == 768 and K == 256,
       "the timing cell must match [R.305]'s own training cell")
    ck(REPEATS >= 3 and REPS >= 20 and WARMUP >= 10,
       "shared-box hygiene needs repeats, reps and warmup")

    # -- 2. EVERY arm in the joint table must be timeable --------------------
    import r307_cost_table as T
    t307 = {ALIAS.get(a["key"], a["key"]) for a in T.ARMS}
    ck(t307 <= set(ARM_ORDER),
       f"[R.307] arms missing from the timing rig: {sorted(t307 - set(ARM_ORDER))}")
    ck(ALIAS["scora2"] == "scora",
       "the two SCoRA rows differ only in lr and scaling -- IDENTICAL forward, so "
       "they share one timing row; timing them twice would imply a difference")
    ck("frozen" in ARM_ORDER, "a frozen-Linear reference is required for marginals")
    ck("scora_factored" in ARM_ORDER,
       "the factored SCoRA path must be timed too -- [R.307] showed the AS-RUN "
       "path is the dense one, so both belong in the wall-clock column")

    # -- 3. the estimator is MIN across repeats, and the control proves it ---
    rows = [dict(arm="a", tf32="1", mode="fwd", rep=str(i), ms=str(m), syncs="0",
                 mem_bytes="") for i, m in enumerate([5.0, 9.0, 7.0])]
    rows += [dict(arm="frozen", tf32="1", mode="fwd", rep="0", ms="4.0", syncs="0",
                  mem_bytes="")]
    e = estimate(rows, 1)
    ck(abs(e["a"]["fwd"][0] - 5.0) < 1e-9,
       f"estimator must be the MIN block-median (5.0), got {e['a']['fwd'][0]}")
    ck(abs(e["a"]["fwd"][1] - 4.0) < 1e-9, "spread must be max-min")
    ck(abs(marginal(rows, 1)["a"]["fwd"] - 1.0) < 1e-9,
       "marginal must subtract the frozen Linear")
    #   CONTROL: a mean estimator would give 7.0 -- if this ever passes, the
    #   shared-box reasoning has been silently dropped.
    ck(abs(statistics.fmean([5.0, 9.0, 7.0]) - e["a"]["fwd"][0]) > 1e-6,
       "the estimator must NOT be the mean")

    # -- 4. tf32 is a real axis, not a label --------------------------------
    rows2 = rows + [dict(arm="a", tf32="0", mode="fwd", rep="0", ms="99.0",
                         syncs="0", mem_bytes="")]
    ck(estimate(rows2, 1)["a"]["fwd"][0] == 5.0, "tf32=1 must not see tf32=0 rows")
    ck(estimate(rows2, 0)["a"]["fwd"][0] == 99.0, "tf32=0 must be readable separately")

    # -- 5. ⛔ an arm that FAILS the sync gate must not be timed -------------
    rows3 = [dict(arm="bad", tf32="1", mode="fwd", rep="-1", ms="", syncs="2",
                  mem_bytes="")]
    ck("bad" not in estimate(rows3, 1),
       "an arm with no timed rows must not appear in the estimate")
    src = open(os.path.join(HERE, "r308_timing.py")).read()
    ck("set_sync_debug_mode" in src and "are NOT timed" in src,
       "the sync gate must remain wired into the run path")
    ck("loca" in src and "asymmetrically-repaired" in src,
       "the rationale for gating the BASELINES must stay in the file")

    # -- 6. the end-to-end cap must be printed, not left to the reader ------
    ck("1.11-1.16x" in src and "10-13%" in src,
       "the report must convert a module-level ratio into its end-to-end cap")

    # -- 7. if a measurement exists, it must be internally consistent -------
    real = load()
    if real:
        e1 = estimate(real, 1)
        ck("frozen" in e1, "a real run must include the frozen reference")
        ck(all(int(r["syncs"]) == 0 for r in real if r["ms"]),
           "⛔ every TIMED row must have passed the 0-sync gate")
        for a, d in e1.items():
            ck(d["fwdbwd"][0] > d["fwd"][0],
               f"{a}: fwd+bwd must exceed fwd, else the backward is not running")

    print(f"[r308] selftest: {n[0]} passed, 0 failed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.run:
        run(a.device)
    else:
        report()
