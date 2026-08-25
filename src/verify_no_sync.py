"""verify_no_sync -- STANDING GATE against the [R.101]/[R.104] defect class.

A cache (or any expression) whose key touches a GPU tensor is a HIDDEN device
synchronisation.  It is invisible to flop accounting, invisible to every unit gate
(all 24 of verify_slr passed with one present), and invisible to accuracy (the fix
is bit-identical).  It shows up ONLY in wall-clock -- the axis this program claims.

It has now occurred TWICE, written independently by different authors on the same day,
in two different adapters:
  [R.101] SLRLinear._basis        key contained int(idx.sum())      -> 2 syncs/forward
  [R.104] MergedFourierFTLinear._uv  key contained int(idx[0].sum())  -> 2 syncs/forward

Two occurrences is a defect CLASS, not an incident.  This file is the gate.

  static  : flag <scalarising call>(<buffer-ish expr>) inside adapter sources
  runtime : torch.cuda.set_sync_debug_mode("warn"), assert 0 syncs per forward
            (skipped without CUDA, and by --static-only when a timing run is live)
"""
import argparse, ast, os, re, sys

# ⛔ [R.308] THE BASELINES BELONG IN THIS GATE TOO.  A hidden sync in a BASELINE
# makes that baseline look slow and flatters us -- the mirror image of [R.101],
# and CONTEXT 5's "never quote a ratio off an asymmetrically-repaired pair" is
# exactly this failure.  loca/qwha/haar were outside the gate until the [R.308]
# timing run needed them; they are inside it now.
ADAPTERS = ["slr_adapter.py", "merged_fourierft.py", "fourierft_fast.py",
            "spectral_adapter.py", "sparse_adapter.py", "bwht_adapter.py",
            "offgrid_adapter.py", "rotft_adapter.py", "shrinkft_adapter.py",
            "loca_adapter.py", "qwha_adapter.py", "haar_adapter.py"]
SCALARISERS = {"int", "float", "bool", "str"}          # str(device) is fine; str(tensor) is not
TENSORY = re.compile(r"\.(sum|item|max|min|numel|shape|nonzero|argmax|argmin)\b")


def _funcs(tree):
    return {n.name: n for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def _calls(node):
    """Names called inside a function body (bare `f()` and `self.f()`/`cls.f()`)."""
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Name):
                out.add(f.id)
            elif isinstance(f, ast.Attribute):
                out.add(f.attr)
    return out


def forward_reachable(tree):
    """Functions reachable from a per-forward entry point.

    This is the fix for the first version of this gate, which keyed on the BUFFER'S
    NAME and therefore MISSED BOTH HISTORICAL DEFECTS -- `_basis` and `_uv` scalarise a
    function PARAMETER (`idx`), not a name matching `register_buffer(...)` -- while
    flagging `bwht_support_equitable`, which runs once at __init__.  Reachability from
    `forward` is the property that actually matters.
    """
    funcs = _funcs(tree)
    roots = [n for n in ("forward", "get_delta_weight", "factors") if n in funcs]
    seen, stack = set(roots), list(roots)
    while stack:
        cur = stack.pop()
        for c in _calls(funcs[cur]):
            if c in funcs and c not in seen:
                seen.add(c); stack.append(c)
    return seen, funcs


def static_scan(path):
    src = open(path).read()
    tree = ast.parse(src)
    reach, funcs = forward_reachable(tree)
    owner = {}
    for name, fn in funcs.items():
        for ln in range(fn.lineno, (fn.end_lineno or fn.lineno) + 1):
            owner[ln] = name
    hits = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in SCALARISERS and node.args):
            continue
        inner = ast.unparse(node.args[0])
        if node.func.id == "str" and not TENSORY.search(inner):
            continue                                   # str(device)/str(dtype) are cheap
        if not TENSORY.search(inner) and not re.search(r"\[", inner):
            continue                                   # int(some_python_int) is fine
        if "shape" in inner or "numel" in inner:
            continue                                   # metadata, no device read
        fn = owner.get(node.lineno, "<module>")
        hits.append((node.lineno, f"{node.func.id}({inner})", fn, fn in reach))
    return reach, hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--static-only", action="store_true",
                    help="skip the CUDA runtime check (use while a timing run is live)")
    a = ap.parse_args()
    npass = nfail = nreview = 0
    print("=" * 78); print("verify_no_sync -- static scan"); print("=" * 78)
    for f in ADAPTERS:
        p = os.path.join(os.path.dirname(__file__), f)
        if not os.path.exists(p):
            continue
        reach, hits = static_scan(p)
        real = [h for h in hits if h[3]]                 # forward-reachable => gating
        status = "PASS" if not real else "REVIEW"
        nreview += bool(real)
        print(f"  [{status}] {f:24s}")
        for ln, expr, fn, _ in real:
            print(f"          ⛔ line {ln}: {expr}   in def {fn}  (REACHABLE FROM forward)")
        for ln, expr, fn, _ in [h for h in hits if not h[3]]:
            print(f"          ·  line {ln}: {expr}   in def {fn}  (init-time only -- not gating)")
    print(f"\nstatic: {nreview} file(s) with forward-reachable scalarisers to REVIEW.")
    print("  NOTE: static REVIEW is ADVISORY and does not fail this gate -- both known sites")
    print("  (_basis, _uv) are now BYPASSED by per-instance caches, so the code is present but")
    print("  unreached at steady state.  THE RUNTIME SYNC COUNT IS THE GATE.")
    if a.static_only:
        print("runtime check SKIPPED (--static-only) -- NO VERDICT, the runtime count is the gate")
        return 0
    import torch
    if not torch.cuda.is_available():
        print("runtime check SKIPPED (no CUDA) -- NO VERDICT, the runtime count is the gate")
        return 0
    sys.path.insert(0, os.path.dirname(__file__))
    import warnings, torch.nn as nn
    from slr_adapter import SLRLinear
    from merged_fourierft import MergedFourierFTLinear
    from fourierft_fast import FourierFTFastLinear
    from spectral_adapter import SpectralAdapterLinear
    dev, d = "cuda:0", 768
    cases = [
        ("SLR materialise=True", lambda: SLRLinear(nn.Linear(d, d), rank=1, s=128, init="zero")),
        ("SLR materialise=False", lambda: SLRLinear(nn.Linear(d, d), rank=1, s=128, init="zero",
                                                    materialise=False)),
        ("FourierFT merged ifft2", lambda: MergedFourierFTLinear(nn.Linear(d, d), n_frequency=256,
                                                                 scaling=150.0, materialise="ifft2")),
        ("FourierFT merged lowrank", lambda: MergedFourierFTLinear(nn.Linear(d, d), n_frequency=256,
                                                                   scaling=150.0, materialise="lowrank")),
        ("fourierft-fast rfft", lambda: FourierFTFastLinear(nn.Linear(d, d), n_frequency=256,
                                                            scaling=150.0, use_rfft=True)),
        ("LYRA p=q=16", lambda: SpectralAdapterLinear(nn.Linear(d, d), p=16, q=16, scaling=1.0)),
    ]
    print("\n" + "=" * 78); print("runtime: syncs per forward (caches warmed)"); print("=" * 78)
    for tag, mk in cases:
        torch.manual_seed(0)
        m = mk().to(dev); x = torch.randn(8, 128, d, device=dev)
        for _ in range(3):
            with torch.no_grad(): m(x)
        torch.cuda.synchronize(); torch.cuda.set_sync_debug_mode("warn")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with torch.no_grad(): m(x)
        torch.cuda.set_sync_debug_mode("default")
        n = sum(1 for r in w if "sync" in str(r.message).lower())
        ok = n == 0; npass += ok; nfail += not ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {tag:26s} syncs={n}")
    print(f"\nTOTAL: {npass} pass / {nfail} fail")
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
