"""R.181 gates -- LoCA integration.  The point of this file is ONE claim:

    our `LoCALinear` computes the AUTHORS' dW, bit-for-bit.

That is the [R.95] discipline applied to a new baseline: every FourierFT number
in this repo is attributable to the official operator because we proved our dW
is bit-identical to `peft.tuners.fourierft`.  A baseline we re-implemented would
be OUR reading of LoCA, and a loss would be unattributable.

The authors' layer is exercised in a SUBPROCESS with ./temp/LoCA/peft/src on
sys.path, so their PEFT fork never enters the process that our code runs in --
swapping PEFT globally would put the FourierFT gate at risk.

Run:  env/bin/python src/verify_loca_adapter.py
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tempfile

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from loca_adapter import LoCALinear, get_loca_adapter_model  # noqa: E402

FAIL = []
def chk(i, name, ok, det=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {i}  {name}" + (f"   {det}" if det else ""))
    if not ok:
        FAIL.append(i)

print("=" * 78); print("  R.181 -- LoCA integration gates"); print("=" * 78)

# --- G0: the vendored DCT utils are byte-identical to the authors' ----------- #
theirs = os.path.join(ROOT, "temp/LoCA/peft/src/peft/tuners/loca/dct_utils.py")
ours = os.path.join(HERE, "loca_dct_utils.py")
if os.path.exists(theirs):
    a = open(theirs, "rb").read()
    b = open(ours, "rb").read()
    # ours = header + theirs
    chk(0, "loca_dct_utils.py is the authors' file verbatim (header prepended only)",
        b.endswith(a), f"their sha1 {hashlib.sha1(a).hexdigest()[:12]}")
else:
    chk(0, "authors' dct_utils.py present at ./temp/LoCA", False, "clone missing")

# --- G1: dW BIT-IDENTICAL to the authors' own layer ------------------------- #
DRIVER = r'''
import sys, torch
sys.path.insert(0, "%s")
from peft.tuners.loca.layer import Linear as TheirLinear
torch.manual_seed(0)
base = torch.nn.Linear(%d, %d, bias=False)
lay = TheirLinear(base, "default", n_frequency=%d, scale=%f,
                  learn_location_iter=100, loca_dct_mode="default")
sd = torch.load("%s")
with torch.no_grad():
    lay.spectrum["default"].copy_(sd["spectrum"])
    lay.spectrum_indices["default"].copy_(sd["indices"])
torch.save({"dW": lay.get_delta_weight("default").detach()}, "%s")
print("OK")
'''

def bit_identity(in_f, out_f, k, scale, seed):
    torch.manual_seed(0)
    base = torch.nn.Linear(in_f, out_f, bias=False)
    ours_layer = LoCALinear(base, n_frequency=k, scale=scale,
                            learn_location_iter=100, init_seed=seed)
    with torch.no_grad():                       # non-trivial spectrum, else dW==0 trivially
        ours_layer.spectrum.normal_(generator=torch.Generator().manual_seed(seed + 1))
    td = tempfile.mkdtemp()
    fin, fout = os.path.join(td, "in.pt"), os.path.join(td, "out.pt")
    torch.save({"spectrum": ours_layer.spectrum.detach(),
                "indices": ours_layer.spectrum_indices.detach()}, fin)
    src = DRIVER % (os.path.join(ROOT, "temp/LoCA/peft/src"), in_f, out_f, k, scale, fin, fout)
    drv = os.path.join(td, "drv.py")
    with open(drv, "w") as fh:
        fh.write(src)
    r = subprocess.run([sys.executable, drv], capture_output=True, text=True, cwd=ROOT)
    if r.returncode != 0:
        return None, r.stderr.strip().splitlines()[-1] if r.stderr else "subprocess failed"
    theirs_dW = torch.load(fout)["dW"]
    ours_dW = ours_layer.get_delta_weight().detach()
    return torch.equal(ours_dW, theirs_dW), f"max|diff| = {(ours_dW - theirs_dW).abs().max():.3e}"

for (m, n, k, sc) in ((768, 768, 256, 1.0), (768, 768, 1000, 1.0), (512, 768, 64, 0.5)):
    ok, det = bit_identity(n, m, k, sc, 777)
    if ok is None:
        chk(1, f"dW bit-identical to authors' layer (d={m}, k={k})", False, det)
    else:
        chk(1, f"dW bit-identical to authors' layer (m={m}, n={n}, k={k}, scale={sc})", ok, det)

# --- G2: zero init => dW == 0 exactly (paper 5) ----------------------------- #
torch.manual_seed(0)
L = LoCALinear(torch.nn.Linear(768, 768, bias=False), n_frequency=256, init_seed=777)
chk(2, "zero coefficient init gives dW == 0 exactly", bool(torch.all(L.get_delta_weight() == 0)))

# --- G3: the alternating schedule matches the paper (Ba=10, Bl=20) ---------- #
L = LoCALinear(torch.nn.Linear(768, 768, bias=False), n_frequency=256,
               learn_location_iter=90, init_seed=777)
L.train()
x = torch.randn(2, 768)
phases = []
for it in range(120):
    L(x)
    phases.append((L.spectrum.requires_grad, L.spectrum_indices.requires_grad))
# after iter 0..9 coefficients; 10..29 locations; repeats; >=90 coefficients only
ok = (phases[0] == (True, False) and phases[9] == (True, False)
      and phases[10] == (False, True) and phases[29] == (False, True)
      and phases[30] == (True, False) and phases[100] == (True, False))
loc_iters = sum(1 for p in phases[:90] if p[1])
chk(3, "alternating schedule = 10 coeff / 20 loc, frozen after learn_location_iter",
    ok, f"{loc_iters}/90 location steps before freeze, none after")

# --- G4: PARAMETER ACCOUNTING -- the [R.180 4.1] flag, made visible --------- #
L = LoCALinear(torch.nn.Linear(768, 768, bias=False), n_frequency=256, init_seed=777)
reported = L.spectrum.numel()
optimised = L.spectrum.numel() + L.spectrum_indices.numel()
chk(4, "reported budget is k; OPTIMISED budget is 3k during the alternating phase",
    reported == 256 and optimised == 768, f"reported {reported}, optimised {optimised} (=3k)")

# --- G5: the model wrapper adapts exactly the target modules ---------------- #
class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Linear(768, 768, bias=False)
        self.key = torch.nn.Linear(768, 768, bias=False)
        self.value = torch.nn.Linear(768, 768, bias=False)
        self.classifier = torch.nn.Linear(768, 2)
    def forward(self, x): return self.classifier(self.value(self.key(self.query(x))))

t = Toy()
wrapped = get_loca_adapter_model(t, ["query", "value"], n_frequency=256, init_seed=777)
n_loca = len(wrapped.loca_layers)
adapted = {n for n, m_ in t.named_modules() if isinstance(m_, LoCALinear)}
chk(5, "wraps exactly query+value, leaves key alone",
    n_loca == 2 and adapted == {"query", "value"}, f"adapted {sorted(adapted)}")

# --- G6: base weights frozen, head trainable ------------------------------- #
base_frozen = all(not p.requires_grad for p in t.query.base_layer.parameters())
head_train = all(p.requires_grad for p in t.classifier.parameters())
chk(6, "base weights frozen, classifier head trainable", base_frozen and head_train)

print("=" * 78)
print(f"{7-len(set(FAIL))}/7 gate groups pass" if not FAIL else f"FAILED: {sorted(set(FAIL))}")
print("=" * 78)
sys.exit(1 if FAIL else 0)
