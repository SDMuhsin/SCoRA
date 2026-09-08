#!/usr/bin/env python3
"""
optimizer_map_gate.py — the optimizer NAME -> CLASS map is unchanged by the lazy-import fix.

⛔ WHY THIS EXISTS (narval, 2026-09-08).
   `import bitsandbytes` dies with SIGILL on narval (EPYC 7532, no avx512f), and
   `galore_torch` imports it transitively. Both were imported AT MODULE SCOPE in
   src/train_glue.py, so the runner could not be imported on that cluster AT ALL --
   for every arm, including the ones using plain torch.optim.AdamW.

   ⚠⚠ A try/except could not fix it. SIGILL is a SIGNAL: it kills the interpreter,
     so `except ImportError` never runs. The import must not execute at all, which
     meant replacing a dict of 20 eagerly-imported classes with a resolver.

   ⛔ THAT DICT IS LOAD-BEARING HISTORY: 41 banked drivers selected their optimizer
     through it. This gate pins the replacement against the mapping READ OUT OF THE
     PRE-CHANGE SOURCE (git), so "the map is unchanged" is a measurement and not a
     claim. It deliberately does NOT import galore_torch or bitsandbytes -- it
     compares the SYMBOL each name resolves to, which is what the old dict stored.

   Run:  python3 scripts/optimizer_map_gate.py
"""
import ast, re, subprocess, sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "train_glue.py"
# The commit before the lazy-import change; the dict is read out of it, never retyped.
BASE = "ce5b3d6"

_ok, _bad = [], []


def check(name, cond, detail=""):
    (_ok if cond else _bad).append(name)
    print(f"  {'✅' if cond else '⛔'} {name}")
    if not cond and detail:
        print("        | " + str(detail)[:600].replace("\n", "\n        | "))


def old_map():
    """Parse the ORIGINAL dict literal out of the pre-change file in git."""
    old = subprocess.run(["git", "show", f"{BASE}:src/train_glue.py"],
                         cwd=ROOT, capture_output=True, text=True).stdout
    if not old:
        return None
    m = re.search(r"optimizer_classes = \{(.*?)\n    \}", old, re.S)
    if not m:
        return None
    out = {}
    # Entries look like  'name': SomeClass  or  'name': mod.attr.Class
    for key, val in re.findall(r"'([a-z0-9_]+)':\s*([A-Za-z_][\w.]*)", m.group(1)):
        out[key] = val
    return out


def new_map():
    """What the resolver returns, expressed as the same symbol names, WITHOUT
    importing any optimizer library: read the resolver's own source."""
    tree = ast.parse(SRC.read_text())
    galore = {}
    fn = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "_GALORE_NAMES":
            galore = ast.literal_eval(node.value)
        if isinstance(node, ast.FunctionDef) and node.name == "_resolve_optimizer_class":
            fn = node
    assert fn is not None, "_resolve_optimizer_class not found"
    out = dict(galore)
    # The non-GaLore branches: `if name == 'x': ... return <expr>`
    src = ast.get_source_segment(SRC.read_text(), fn)
    for key, ret in re.findall(r"if name == '([a-z0-9_]+)':\s*\n(?:\s*(?:import|from)[^\n]*\n)?\s*return ([\w.]+)", src):
        out[key] = ret
    return out


def main():
    print("=" * 78)
    print("OPTIMIZER MAP GATE — the lazy import changed no name->class mapping")
    print("=" * 78)
    old, new = old_map(), new_map()
    check(f"the pre-change dict is readable out of git ({BASE})",
          old is not None and len(old) >= 19, f"got {old}")
    if old is None:
        print("\n⛔ cannot compare; is the base commit present?")
        return 1

    # ⭐ The comparison, both directions.
    check("every OLD name still exists in the resolver",
          set(old) <= set(new), f"missing: {sorted(set(old) - set(new))}")
    check("⛔ CONTROL: and the resolver invents no new names",
          set(new) <= set(old), f"extra: {sorted(set(new) - set(old))}")
    for k in sorted(old):
        want, got = old[k], new.get(k)
        # bnb.optim.Adam8bit was written as an attribute path; the resolver keeps it.
        check(f"'{k}' -> {want}", got == want, f"resolver gives {got!r}")

    # ⛔ THE CONTROL THAT MATTERS FOR NARVAL: importing the runner must not import
    #   bitsandbytes or galore_torch. That is the entire point of the change.
    src = SRC.read_text()
    top = src[:src.index("def optimizer_names(")] if "def optimizer_names(" in src else src
    check("⭐ neither package is imported at MODULE SCOPE any more",
          "\nimport bitsandbytes" not in top and "\nfrom galore_torch" not in top,
          top[-400:])
    tree = ast.parse(src)
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] + ([node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            if any(n and ("bitsandbytes" in n or "galore_torch" in n) for n in names):
                # It is fine INSIDE a function; fatal at module scope.
                bad.append(node.lineno)
    modlevel = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)) and node.lineno in bad:
            modlevel.append(node.lineno)
    check("⛔ CONTROL: ...checked structurally (ast), not just by text search",
          not modlevel, f"module-scope imports at lines {modlevel}")
    check("...and they ARE still imported somewhere (lazily), so the optimizers work",
          bool(bad), "no import of either package survives — galore arms would break")

    print("\n" + "=" * 78)
    print(f"optimizer_map_gate: {len(_ok)} passed, {len(_bad)} FAILED")
    for b in _bad:
        print(f"  ⛔ {b}")
    return 1 if _bad else 0


if __name__ == "__main__":
    sys.exit(main())
