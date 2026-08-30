#!/usr/bin/env python
"""[R.307] THE JOINT TABLE -- task metric AND computational cost, every arm at its
own tuned optimum.

  env/bin/python scripts/r307_cost_table.py --selftest    # before believing it
  env/bin/python scripts/r307_cost_table.py

Accuracy comes from `[R.305]`/`[R.306]`'s readers (computed, never re-typed).
Cost comes from three MEASURED or DERIVED sources, and each column names which:

  params  [measured]  parsed from the confirmed run's own training log -- both the
                      REPORTED budget and what the optimiser actually updated.
  flops   [derived]   `src/bench_adapter_cost.theoretical_flops`, the frozen J.2
                      op-counter, at the SAME batch the accuracy was measured at
                      (b = 32 x 128 = 4096 tokens), per adapted module.
  memory  [measured]  peak allocator bytes from the confirmed runs themselves.

⛔ WHAT THIS FILE REFUSES TO PRINT
  1. WALL-CLOCK.  `[R.101]`'s "4.8% slower" is WITHDRAWN (two forced GPU syncs,
     since fixed) and CONTEXT 5 bars any replacement until `[R.103]` lands.  The
     `avg_step_time` column in the [R.305]/[R.306] CSVs is NOT a substitute: those
     cells ran THREE-WAY CONCURRENT on a shared box and `[R.209]` measured ~2.2x
     inflation from co-tenancy.  A throughput number from them would be fiction.
  2. A FLOP FIGURE FROM A PATH THE ACCURACY WAS NOT MEASURED ON.  ⭐ Every SCoRA
     accuracy cell in [R.305]/[R.306] ran `--slr_materialise 1` (the flag defaults
     to 1 and neither grid passed it), i.e. the DENSE Theta(b*m*n) path -- not the
     factored Theta(b*d*r) path whose 3,936 flops/token the program quotes.  Both
     are printed, the as-run one first, and the floor is labelled with the flag
     needed to reach it.  `src/verify_slr.py` G6 is what licenses transferring the
     accuracy across the two (max rel diff 1.9e-07, same object).
  3. A COST RATIO WITHOUT ITS REFERENCE.  CONTEXT 5: every cost claim is reported
     against BOTH the dense frozen GEMM and `fourierft-fast`, at matched dtype.

⛔ Read the FLOOR column as a floor.  It is what the arm's own published forward
costs when implemented at its best; it is not what this repo's grid ran.
"""
import argparse, math, os, re, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
import r305_plan as P
import r305_read as R5
import r306_plan as R6
import r306_read as R6r
import bench_adapter_cost as BC

# ---- the cell the accuracy claim is made at, so the cost is quoted there ----
D_MODEL   = 768
K         = 256
BATCH_SEQ = 32
SEQ_LEN   = 128
B_TOKENS  = BATCH_SEQ * SEQ_LEN          # 4096 -- [R.305]'s own training batch
N_MODULES = 24                           # query+value x 12 layers
HEAD_PARAMS = 592_130                    # [P.16] RoBERTa-base classification head


def _f(arm, **kw):
    """adapter-only forward flops for ONE module over B_TOKENS, per token."""
    return BC.theoretical_flops(arm, D_MODEL, D_MODEL, B_TOKENS, K,
                                **kw)["fwd_adapter"] / B_TOKENS


# ============================================================================
# ⭐ THE MERGED FLOOR -- the column that makes this comparison fair.
#
# `y = x (W0 + dW)^T` folds the adapter into a GEMM the model already does, so
# the marginal adapter cost is the dW REBUILD, paid ONCE per forward and
# amortised over the batch (`src/merged_fourierft.py`, [P.5/P.6]).  At b=4096
# that amortisation is 4096x and it REORDERS the family -- so quoting only the
# as-run column would flatter the two arms this repo happens to run merged
# (FourierFT-merged and SCoRA) against six that it does not.
#
# ⛔ Only `fftm` and the two SCoRA rows ACTUALLY RAN merged.  Every other arm's
# merged figure is a floor this repo has never measured end-to-end.  The rule
# below is applied IDENTICALLY to all of them; that is the whole point.
# ============================================================================
def _wht(L):
    return L * math.log2(L)          # additions-only, as in bench_adapter_cost


def _build_flops(key, d=None):
    """Flops to REBUILD the dense m x n dW once, per arm, from its own params.

    ⭐ `d` DEFAULTS to this table's own backbone width but is a PARAMETER, because
      the gemma-2b final runs quote the same op-count at d=2048 and the arm ->
      op-counter mapping must exist in ONE place. A second copy of this map that
      agreed today is exactly the defect class this repo keeps paying for."""
    d = D_MODEL if d is None else d
    if key in ("fftm", "fftstock"):
        return BC.FFT_C(d) * d * 2 + d * d          # ifft2 over rows+cols, then scale
    if key.startswith("wave"):
        mu = 2 if key == "wave2" else 1
        r = BC._haar_tail(d)
        return d * (3 * d - 2 * r) * 2 + 2 * mu * K  # 2-D inverse Haar + core
    if key == "loca":
        return 2 * K * d * d                        # k rank-1 DCT outer products
    if key == "qwha":
        # ⚠️ CONSTRUCTED, not QWHA's algorithm: QWHA never builds dW -- it works
        # in the WHT domain end-to-end.  This is what merging WOULD cost if it
        # did, included so the column is uniform.  Flagged in the report.
        return d * _wht(d) * 2 + 2 * K              # 2-D inverse WHT + sparse add
    if key == "lyra":
        p_ = q_ = max(1, int(round(K ** 0.5)))
        return 2 * d * p_ * q_ + 2 * d * q_ * d     # U D, then (UD) V^T
    if key.startswith("scora"):
        s_ = t_ = K // 2
        return 2 * d * s_ + 2 * d * t_ + 2 * d * d  # rebuild u,v then u v^T
    raise KeyError(key)


def _merged_per_token(key, d=None, b=None):
    """Rebuild + fold-in, amortised over the batch the accuracy was measured at."""
    d = D_MODEL if d is None else d
    b = B_TOKENS if b is None else b
    return (_build_flops(key, d) + d * d) / b


def _crossover(key):
    """Batch size above which merging beats this arm's own unmerged path.

    Reported because the merged column is batch-dependent BY CONSTRUCTION and a
    single-batch number would hide that.  Returns None if merging never wins."""
    build = _build_flops(key) + D_MODEL * D_MODEL
    for b in range(1, 1 << 20):
        un = _unmerged_at(key, b)
        if un is None:
            return None
        if build / b <= un:
            return b
    return None


DENSE_GEMM_PER_TOKEN = BC.GEMM(1, D_MODEL, D_MODEL)    # 2mn, the frozen layer


def _unmerged_at(key, b, d=None):
    """The arm's own UNMERGED forward, per token, at batch `b`.

    This is the path each method publishes, except where this repo's own
    implementation is dearer -- and where it is, the AS-RUN column says so.
    ⭐ `d` is a PARAMETER for the same reason `_build_flops`'s is: ONE arm ->
      op-counter map, quoted at whatever width the accuracy was measured on."""
    m = n = D_MODEL if d is None else d
    if key in ("fftm", "fftstock"):
        a = "fourierft_stock"
        kw = {}
    elif key == "loca":
        a, kw = "loca_stock", {}
    elif key == "qwha":
        a, kw = "qwha", {}
    elif key.startswith("wave"):
        a, kw = "waveft_factored", {"r": 2 if key == "wave2" else 1}
    elif key == "lyra":
        a, kw = "lyra_factored", {}
    elif key.startswith("scora"):
        a, kw = "slr_factored", {}
    else:
        return None
    return BC.theoretical_flops(a, m, n, b, K, **kw)["fwd_adapter"] / b


# ---- the arms, in the order the accuracy table ranks them ------------------
# ⭐ `ran_merged` is the load-bearing field: it records whether [R.305]/[R.306]
# ACTUALLY folded this arm's dW into the frozen GEMM.  Only three rows did, and
# both of ours are among them -- so the as-run column MUST be read next to the
# uniform merged column, never on its own.
ARMS = [
    dict(key="fftm", src="r305", title="FourierFT (merged)", ran_merged=True,
         path="merged: dense ifft2 rebuild/step",
         floor=lambda: _f("fft_fast_rfft"), floor_note="fourierft-fast (rfft, [P.4])"),
    dict(key="fftstock", src="r305", title="FourierFT (stock PEFT)", ran_merged=False,
         path="unmerged: dense ifft2 + 2nd GEMM",
         floor=lambda: _f("fft_fast_rfft"), floor_note="fourierft-fast (rfft, [P.4])"),
    dict(key="loca", src="r305", title="LoCA", ran_merged=False,
         path="builds dense dW, then a 2nd GEMM",
         floor=lambda: _f("loca_factored"), floor_note="factored idct apply, no dW"),
    dict(key="qwha", src="r305", title="QWHA", ran_merged=False,
         path="WHT(W0) + WHT(x), no dW built", floor=None),
    dict(key="scora2", src="r306", title="SCoRA (ours, scaling SWEPT)", ran_merged=True,
         path="⛔ --slr_materialise 1 (default)", floor=None),
    dict(key="scora", src="r305", title="SCoRA (ours, a-priori)", ran_merged=True,
         path="⛔ --slr_materialise 1 (default)", floor=None),
    dict(key="wave2", src="r305", title="WaveFT mu=2 (repo fix)", ran_merged=False,
         path="factored Haar pyramid, no dW", floor=None),
    dict(key="wave1", src="r305", title="WaveFT mu=1 (published)", ran_merged=False,
         path="factored Haar pyramid, no dW", floor=None),
    dict(key="lyra", src="r305", title="LYRA", ran_merged=False,
         path="factored DCT projections", floor=None),
]

STATE = {"r305": R6.R305_D, "r306": R6.D}

# ============================================================================
# MEASURED WALL-CLOCK AND ACTIVATION MEMORY -- [R.308]
# `fftstock` is not timed (PEFT's own layer carries 1 device sync/forward and is
# third-party); FourierFT is represented by the two sync-free implementations
# this repo owns.  `scora2` shares `scora`'s timing: identical forward.
# ============================================================================
import r308_timing as T8

# ⛔ `fftstock` gets NO timing alias.  PEFT's own layer carries 1 device sync per
# forward and is third-party, so it is not timed -- and lending it
# `fourierft-fast`'s numbers would put one implementation's latency on another
# implementation's row.  It shows `n/t`; the sync-free reference is printed below
# the table with its own measured figures.
TIMING_ALIAS = {"scora2": "scora"}
_T8ROWS = T8.load()
TIM_EST = T8.estimate(_T8ROWS, tf32=1)
TIM_MARG = T8.marginal(_T8ROWS, tf32=1)
TIM_MEM = {r["arm"]: float(r["mem_bytes"]) / 1024 ** 2
           for r in _T8ROWS if r["mem_bytes"] and int(r["tf32"]) == 1}



# ============================================================================
# MEASURED COLUMNS -- read from the confirmed runs, never assumed
# ============================================================================
def _confirmed(key, src):
    """(label, mean, sd, n, per_seed, candidate_label) for one arm."""
    if src == "r306":
        rec = R6r.r306_data()[0].get("scora2")
    else:
        rec = R6r.r305_confirmed()[1].get(key)
    if rec is None:
        return None
    return R5.tuned(rec)


def d_labels(key, src, cand):
    """The stage-D labels that ran `cand`, one per confirmation seed."""
    old = P.D
    try:
        P.D = STATE[src]
        man = P.read_manifest()
        return sorted(l for l, m in man.items()
                      if m.get("stage") == "D" and m.get("from") == cand)
    finally:
        P.D = old


TRAINABLE_RE = re.compile(r"trainable params:\s*([\d,]+)")
OPTIMISED_RE = re.compile(r"OPTIMISED during the alternating phase\s*=\s*([\d,]+)")


def params_from_log(src, label):
    """(reported_adapter, optimised_adapter) -- [measured], from the run's log.

    ⭐ LoCA logs both because they DIFFER: it reports 6,144 coefficients but the
    optimiser also updates 2 location coordinates per atom (3x).  Reporting only
    the coefficient count would put it at a parameter parity it does not have.
    """
    path = os.path.join(STATE[src], "logs", label + ".log")
    if not os.path.exists(path):
        return None, None
    reported = optimised = None
    with open(path, errors="replace") as f:
        for line in f:
            m = TRAINABLE_RE.search(line)
            if m and reported is None:
                reported = int(m.group(1).replace(",", "")) - HEAD_PARAMS
            m = OPTIMISED_RE.search(line)
            if m:
                optimised = int(m.group(1).replace(",", ""))
    if optimised is None:
        optimised = reported
    elif reported is not None:
        # the trainable-params line ALREADY counts the extra tensors; the
        # reported budget is the coefficient count the paper quotes.
        reported = optimised // 3 if optimised == reported else reported
    return reported, optimised


def peak_mem(src, labels):
    """Median peak allocator MiB over the confirmed seeds -- [measured]."""
    import csv as _csv
    vals = []
    for lab in labels:
        p = os.path.join(STATE[src], "csv", lab + ".csv")
        if not os.path.exists(p):
            continue
        with open(p) as f:
            rows = list(_csv.DictReader(f))
        if rows and rows[-1].get("peak_mem_mib"):
            try:
                vals.append(float(rows[-1]["peak_mem_mib"]))
            except ValueError:
                pass
    return statistics.median(vals) if vals else None


# ============================================================================
def collect():
    out = []
    for a in ARMS:
        t = _confirmed(a["key"], a["src"])
        if not t:
            continue
        lab, mu, sd, n, vals = t
        labs = d_labels(a["key"], a["src"], lab)
        rep, opt = params_from_log(a["src"], labs[0] if labs else "")
        key = a["key"]
        unmerged = _unmerged_at(key, B_TOKENS)
        merged = _merged_per_token(key)
        t_key = TIMING_ALIAS.get(key, key)
        out.append(dict(a, mean=mu, sd=sd, vals=vals, cand=lab, rep=rep, opt=opt,
                        peak=peak_mem(a["src"], labs),
                        unmerged=unmerged, merged=merged,
                        asrun=merged if a["ran_merged"] else unmerged,
                        cross=_crossover(key),
                        floor_v=a["floor"]() if a["floor"] else None,
                        # ---- storage, DERIVED from the measured param counts --
                        ckpt_kib=(opt or 0) * N_MODULES * 4 / 1024.0,
                        optim_kib=(opt or 0) * N_MODULES * 4 * 2 / 1024.0,
                        # ---- measured wall-clock and activation memory --------
                        t_fwd=TIM_MARG.get(t_key, {}).get("fwd"),
                        t_step=TIM_MARG.get(t_key, {}).get("fwdbwd"),
                        t_xfrozen=(TIM_EST.get(t_key, {}).get("fwdbwd", (None,))[0]
                                   / TIM_EST["frozen"]["fwdbwd"][0]
                                   if t_key in TIM_EST and "frozen" in TIM_EST else None),
                        act_mib=TIM_MEM.get(t_key)))
    return out


def _fmt_ratio(v):
    return f"{v:7.4f}x" if v >= 1e-3 else f"{v:.1e}x"


def report():
    rows = collect()
    if not rows:
        print("[r307] no confirmed rows yet")
        return

    W = 118
    print("=" * W)
    print("[R.307] TASK + COMPUTE, EVERY ARM AT ITS OWN TUNED OPTIMUM")
    print(f"RTE / roberta-base / query+value / k={K} / {N_MODULES} adapted modules / "
          f"batch {BATCH_SEQ}x{SEQ_LEN} = {B_TOKENS} tokens / float32 / TF32 on")
    print("=" * W)
    print(f"{'':30s} {'TASK':>15s}  {'PARAMETERS':>17s}  {'MEMORY':>18s}  {'LATENCY (ms/module)':>21s} {'COMPUTE':>9s}")
    print(f"{'method':30s} {'acc':>7s} {'sd':>7s}  {'train':>8s} {'ckpt':>7s}  "
          f"{'optim':>7s} {'act':>6s} {'peak':>5s}  {'infer':>6s} {'train':>6s} "
          f"{'vs frz':>6s} {'kflop/tk':>9s}")
    print(f"{'':30s} {'':>7s} {'':>7s}  {'params':>8s} {'KiB':>7s}  "
          f"{'KiB':>7s} {'MiB':>6s} {'MiB':>5s}  {'fwd':>6s} {'+bwd':>6s} {'':>6s} {'':>9s}")
    print("-" * W)
    for r in rows:
        n_par = r["opt"] if r["opt"] is not None else r["rep"]
        star = "*" if r["opt"] and r["rep"] and r["opt"] != r["rep"] else " "
        t_f = f"{r['t_fwd']:.3f}" if r["t_fwd"] is not None else "  n/t"
        t_s = f"{r['t_step']:.3f}" if r["t_step"] is not None else "  n/t"
        xf = f"{r['t_xfrozen']:.2f}x" if r["t_xfrozen"] else "   -"
        am = f"{r['act_mib']:.2f}" if r["act_mib"] is not None else "   -"
        pk = f"{r['peak']:.0f}" if r["peak"] else "  -"
        print(f"{r['title']:30s} {r['mean']:7.4f} {r['sd']:7.4f}  "
              f"{n_par:>7,}{star} {r['ckpt_kib']:7.0f}  {r['optim_kib']:7.0f} "
              f"{am:>6s} {pk:>5s}  {t_f:>6s} {t_s:>6s} {xf:>6s} {r['asrun']/1e3:9.2f}")
    print("-" * W)
    fr = TIM_EST.get("frozen", {})
    if fr:
        print(f"{'frozen nn.Linear (reference)':30s} {'-':>7s} {'-':>7s}  {'0':>7s}  "
              f"{'0':>7s} {'0':>7s} {'0.00':>6s} {'-':>5s}  "
              f"{fr['fwd'][0]:6.3f} {fr['fwdbwd'][0]:6.3f} {'1.00x':>6s} "
              f"{DENSE_GEMM_PER_TOKEN/1e3:9.2f}")
    print("\n  train params: what the OPTIMISER updates.  * = differs from the budget the")
    print("    method REPORTS (LoCA reports 6,144 and optimises 3x that: 2 learned")
    print("    location coordinates per atom).  ckpt/optim: 24 modules, fp32, AdamW m+v.")
    print("  act MiB: MEASURED marginal stashed activation memory per module over a frozen")
    print("    Linear.  peak MiB: MEASURED end-to-end training peak (backbone-dominated).")
    print("  latency: MEASURED [R.308], marginal over a frozen Linear, min block-median of")
    print("    5 sweeps x 50 reps, every arm sync-audited to 0 device syncs per forward.")
    print("  kflop/tk: DERIVED, the path each arm actually ran.  n/t = not timed.")
    ff = TIM_MARG.get("fftfast", {})
    if ff:
        print(f"  ⛔ FourierFT (stock PEFT) is NOT timed: its vendored forward carries 1")
        print(f"     device sync.  The repo's sync-free reference implementation")
        print(f"     `fourierft-fast` measures {ff['fwd']:.3f} / {ff['fwdbwd']:.3f} ms "
              f"(fwd / +bwd), {TIM_MEM.get('fftfast', float('nan')):.2f} MiB act.")
        print(f"     QWHA's vendored forward carried 2 syncs and WAS repaired "
              f"bit-identically\n     ([R.308], gate G8) -- otherwise this table would "
              f"time a repaired arm of ours\n     against an unrepaired baseline.")

    # ---- ⭐ the headline: flops and wall-clock DISAGREE -------------------
    timed = [r for r in rows if r["t_step"] is not None]
    if timed:
        by_flop = " < ".join(r["key"] for r in sorted(timed, key=lambda r: r["asrun"]))
        by_time = " < ".join(r["key"] for r in sorted(timed, key=lambda r: r["t_step"]))
        print(f"\n⭐ cheapest-first, and FLOPS AND WALL-CLOCK DISAGREE ([P.2], dispatch-bound):")
        print(f"   by flops     : {by_flop}")
        print(f"   by train time: {by_time}")
        print("   ⛔ Reading either column as 'the cost' is wrong.  WaveFT is 2nd-cheapest")
        print("      in flops and near-LAST in wall-clock; LYRA is 2nd-DEAREST in flops and")
        print("      2nd-fastest.  Op-counts do not predict this family's runtime.")

    # ---- SCoRA's position, on each axis, with the conversion --------------
    ours = next((r for r in rows if r["key"] == "scora"), None)
    pub = [r for r in rows if not r["key"].startswith("scora")]
    if ours and pub:
        fast = min((r for r in pub if r["t_step"]), key=lambda r: r["t_step"])
        lean = min((r for r in pub if r["act_mib"] is not None),
                   key=lambda r: r["act_mib"])
        # ⛔ compare MERGED-to-MERGED.  `asrun` mixes conventions (only 3 arms ran
        # merged, two of them ours), so an as-run flop ratio would flatter us.
        cheap = min(pub, key=lambda r: r["merged"])
        print(f"\nSCoRA vs the BEST PUBLISHED ARM on each axis (not the same arm each time):")
        print(f"   accuracy       {ours['mean']:.4f} vs {max(pub, key=lambda r: r['mean'])['mean']:.4f} "
              f"({max(pub, key=lambda r: r['mean'])['key']})   ⛔ LOSES")
        print(f"   train latency  {ours['t_step']:.3f} vs {fast['t_step']:.3f} ms ({fast['key']})"
              f"   = {fast['t_step']/ours['t_step']:.2f}x faster  ⭐ WINS")
        print(f"   flops/token    {ours['merged']/1e3:.2f} vs {cheap['merged']/1e3:.2f} k ({cheap['key']})"
              f"   = {cheap['merged']/ours['merged']:.2f}x fewer    ⭐ WINS")
        print(f"                  (merged-vs-merged, the UNIFORM rule; unmerged-vs-"
              f"unmerged it is {min(r['unmerged'] for r in pub)/ours['unmerged']:.2f}x)")
        print(f"   activation mem {ours['act_mib']:.2f} vs {lean['act_mib']:.3f} MiB ({lean['key']})"
              f"  = {ours['act_mib']/max(lean['act_mib'],1e-9):.0f}x MORE   ⛔ LOSES")
        print(f"   parameters     {ours['opt']:,} vs {min(r['opt'] for r in pub):,}"
              f"                 = parity")
        gain = fast["t_step"] / ours["t_step"]
        e2e = 1 / (1 - 0.13 * (1 - 1 / gain))
        print(f"\n⛔ THE LATENCY WIN, CONVERTED: a {gain:.2f}x module-level win is worth at most")
        print(f"   {e2e:.3f}x of a training run -- the adapter is 10-13% of wall-clock [P.5-P.11].")
        fl = next((r for r in rows if r["key"] == "scora_factored"), None)
        print(f"   ⭐ AND THE TWO SCoRA PATHS TRADE OFF: --slr_materialise 1 (as run) is the")
        print(f"      FASTEST arm but stashes {ours['act_mib']:.2f} MiB/module; --slr_materialise 0")
        print(f"      stashes {TIM_MEM.get('scora_factored', float('nan')):.3f} MiB "
              f"({ours['act_mib']/max(TIM_MEM.get('scora_factored',1e-9),1e-9):.0f}x leaner) and costs "
              f"{TIM_MARG.get('scora_factored',{}).get('fwdbwd',float('nan'))/ours['t_step']:.1f}x the time.")

    # ---- what the table still does not say --------------------------------
    print("\n" + "=" * W)
    print("⛔ WHAT THIS TABLE STILL DOES NOT SHOW")
    print("=" * W)
    print("* AN END-TO-END SPEEDUP.  Every latency here is a MODULE marginal.  The adapter")
    print("  is 10-13% of training wall-clock [P.5-P.11] -> any win is capped at 1.11-1.16x,")
    print("  and merged training already consumes ~1.06-1.11x of that ceiling.")
    print("* A RESULT AT ANOTHER SCALE.  d=768 only.  Below d ~ 3000 the dense GEMM on")
    print("  tensor cores beats every frequency-domain path, so these ratios are specific")
    print("  to RoBERTa-base; TF32-off numbers are in `r308_timing.py` and differ.")
    print("* A STABLE NUMBER FOR EVERY ARM.  The across-sweep spread is <0.01 ms for")
    print("  fourierft-fast but 3.9-4.2 ms for WaveFT and QWHA on a shared box; the")
    print("  estimator is the MIN block-median and the spread is printed by r308_timing.py.")
    print("* AN ACCURACY EXCUSE.  SCoRA loses the task metric by 0.0325 at the median")
    print("  ([R.305]/[R.306]) and no cost column changes that.")


# ============================================================================
def selftest():
    n = [0]

    def ck(c, m):
        n[0] += 1
        if not c:
            print("FAIL:", m); sys.exit(1)

    # -- 1. the reference cell is the one the ACCURACY was measured at -------
    ck(B_TOKENS == 4096, "b must be 32x128, [R.305]'s own training batch")
    ck(abs(DENSE_GEMM_PER_TOKEN - 2 * 768 * 768) < 1e-9, "dense GEMM ref is 2mn")

    # -- 2. the counter reproduces [R.165]'s BANKED per-token flops ---------
    #    The one external check available: a drift in bench_adapter_cost shows
    #    up here rather than silently re-ranking the family.
    banked = {"mu1": 6644, "mu2": 7156, "slr": 3936, "fft": 39878}
    got = {"mu1": _f("waveft_factored", r=1), "mu2": _f("waveft_factored", r=2),
           "slr": _f("slr_factored"), "fft": _f("fft_fast_rfft")}
    for nm in banked:
        ck(abs(got[nm] - banked[nm]) / banked[nm] < 0.02,
           f"[R.165] banked {nm} = {banked[nm]}; counter now says {got[nm]:.0f}")
    ck(abs(got["mu1"] / got["slr"] - 1.69) < 0.02,
       f"CONTEXT 5's 1.69x must reproduce, got {got['mu1']/got['slr']:.3f}")

    # -- 3. ⛔ THE MERGED COLUMN MUST BE THE SAME RULE FOR EVERY ARM --------
    #    [measured, this file] merging reorders the family at b=4096, and only
    #    3 of 9 rows actually ran merged -- two of them OURS.  If the merged
    #    figure were computed only for our arm, the table would be rigged.
    for a in ARMS:
        v = _merged_per_token(a["key"])
        ck(v > 0, f"{a['key']} must have a merged figure computed by the shared rule")
        ck(abs(v - (_build_flops(a["key"]) + D_MODEL ** 2) / B_TOKENS) < 1e-9,
           f"{a['key']}'s merged figure must come from the SHARED formula")
    ck(len({a["key"] for a in ARMS if a["ran_merged"]}) == 3,
       "exactly 3 arms ran merged (fftm + both SCoRA rows)")
    ck(all(a["ran_merged"] for a in ARMS if a["key"].startswith("scora")),
       "both SCoRA rows ran merged (--slr_materialise defaults to 1)")

    # -- 4. ⭐ THE RANKING FLIP IS REAL, and the table must expose it -------
    rows_key = [a["key"] for a in ARMS]
    o_un = sorted(rows_key, key=lambda k: _unmerged_at(k, B_TOKENS))
    o_me = sorted(rows_key, key=lambda k: _merged_per_token(k))
    o_run = sorted(rows_key, key=lambda k: (_merged_per_token(k)
                   if [a for a in ARMS if a["key"] == k][0]["ran_merged"]
                   else _unmerged_at(k, B_TOKENS)))
    # [measured, this file] the two UNIFORM rules agree on the order -- so the
    # flop ranking is a property of the METHODS, not of the merge choice.  The
    # AS-RUN order does not agree, because it mixes the two conventions.  That
    # is exactly why the as-run column may never be quoted on its own.
    ck(o_un == o_me or [k for k in o_un if not k.startswith("wave")]
       == [k for k in o_me if not k.startswith("wave")],
       f"the two uniform rules must agree on order (wave1/wave2 tie when merged):"
       f"\n   unmerged {o_un}\n   merged   {o_me}")
    ck(o_run != o_un,
       "the AS-RUN order must DIFFER from the uniform ones -- if it does not, "
       "this control is stale and the three-column presentation is unnecessary")
    ck(o_me[0].startswith("scora") and o_un[0].startswith("scora"),
       f"SCoRA must be cheapest under BOTH uniform rules or the claim is not "
       f"convention-independent: merged {o_me[0]}, unmerged {o_un[0]}")
    #   CONTROL: our own as-run figure must NOT be the cheapest by default --
    #   if it is, say so, but never by comparing merged-us to unmerged-them.
    ck(_merged_per_token("scora") < _unmerged_at("scora", B_TOKENS),
       "at b=4096 merging must beat SCoRA's own factored path (it amortises)")
    ck(_crossover("scora") is not None and 128 < _crossover("scora") < 1024,
       f"SCoRA's merge crossover must be a real batch threshold, got {_crossover('scora')}")

    # -- 5. params: LoCA is NOT at parameter parity -------------------------
    rows = collect()
    if rows:
        by = {r["key"]: r for r in rows}
        ck(all(r["rep"] == 6144 for r in rows if r["key"] != "loca" and r["rep"]),
           f"every non-LoCA arm must report 6,144: { {r['key']: r['rep'] for r in rows} }")
        ck(by["loca"]["opt"] == 3 * 6144,
           f"LoCA optimises 3x its reported budget, got {by['loca']['opt']}")
        # -- 6. accuracy must come from the READERS, not be re-typed --------
        ck(abs(by["fftm"]["mean"] - 0.7906) < 5e-4, "fftm must read 0.7906")
        ck(abs(by["scora"]["mean"] - 0.7480) < 5e-4, "scora must read 0.7480")
        ck(abs(by["scora2"]["mean"] - 0.7552) < 5e-4, "scora2 must read 0.7552")
        ck(all(len(r["vals"]) == len(P.CONFIRM_SEEDS) for r in rows), "5 seeds per row")
        ck(all(r["peak"] for r in rows), "every arm must carry measured peak memory")

    # -- 6b. ⛔ MEASURED LATENCY MUST NOT BE BORROWED ACROSS IMPLEMENTATIONS
    ck("fftstock" not in TIMING_ALIAS,
       "PEFT's stock layer must NOT inherit fourierft-fast's timing -- different "
       "implementation, and it was not timed because it carries a device sync")
    ck(TIMING_ALIAS.get("scora2") == "scora",
       "the two SCoRA rows share one timing row: identical forward")
    if rows:
        by = {r["key"]: r for r in rows}
        ck(by["fftstock"]["t_step"] is None, "fftstock must have NO timing")
        ck(by["scora"]["t_step"] == by["scora2"]["t_step"],
           "the two SCoRA rows must carry the SAME measured latency")
        ck(all(r["t_step"] > 0 for r in rows if r["t_step"] is not None),
           "a timed arm must have positive marginal latency")
        # ⭐ the finding this table exists to make visible
        ck(sorted((r["key"] for r in rows if r["t_step"]), key=lambda k: by[k]["asrun"])
           != sorted((r["key"] for r in rows if r["t_step"]), key=lambda k: by[k]["t_step"]),
           "flops and wall-clock must give DIFFERENT orders -- if they ever agree, "
           "[P.2]'s dispatch-bound finding has changed and this table's framing is stale")
        # storage columns are derived from the MEASURED param counts, not assumed
        ck(abs(by["loca"]["ckpt_kib"] - 18432 * 24 * 4 / 1024) < 1e-6,
           "LoCA's checkpoint must be computed from its OPTIMISED param count")
        ck(by["loca"]["ckpt_kib"] == 3 * by["scora"]["ckpt_kib"],
           "LoCA's checkpoint must be 3x ours, as its param count is")

    # -- 7. ⛔ the refusals must survive edits to the report ----------------
    src = open(os.path.join(HERE, "r307_cost_table.py")).read()
    ck("WITHDRAWN" in src and "R.103" in src, "the wall-clock refusal must stay")
    ck("avg_step_time" in src and "2.2x" in src,
       "the report must say WHY the CSVs' step time is not a substitute")
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report()
    out = buf.getvalue()
    if "TASK + COMPUTE" in out:
        ck("WALL-CLOCK" in out, "the report must print the wall-clock refusal")
        ck("FLOPS AND WALL-CLOCK DISAGREE" in out,
           "the report must expose that op-counts do not predict runtime [P.2]")
        ck("n/t" in out, "an arm that was not timed must show n/t, never a borrowed number")
        ck("merged-vs-merged" in out,
           "the flop comparison must state that it is convention-uniform")
        ck("LOSES" in out and "WINS" in out,
           "the per-axis verdict must name losses as plainly as wins")
        ck("dense GEMM" in out and "fourierft-fast" in out,
           "CONTEXT 5 requires BOTH references to be printed")
        ck("dispatch-bound" in out.lower(),
           "the report must say flops do not convert [P.2]")
        ck("1.11-1.16x" in out, "the report must state the end-to-end ceiling")

    print(f"[r307] selftest: {n[0]} passed, 0 failed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    selftest() if a.selftest else report()
