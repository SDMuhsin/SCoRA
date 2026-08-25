"""R.187 gates -- QWHA integration.  The claim under test is the [R.95] one:

    our QWHALinear computes the AUTHORS' FORWARD, bit-for-bit.

Note we gate the FORWARD OUTPUT, not `get_delta_weight` -- because QWHA's
`get_delta_weight` returns a sparse SPECTRUM, not a dW, and the method lives in
how that spectrum is combined with wht(W0) and wht(x).

Their layer runs in a SUBPROCESS with ./temp/qwha/peft/src on sys.path, so their
PEFT fork never enters the process our code runs in ([R.95] protects the FourierFT arm).

Run:  env/bin/python src/verify_qwha_adapter.py
"""
from __future__ import annotations

import hashlib, os, subprocess, sys, tempfile
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from qwha_adapter import QWHALinear, get_qwha_adapter_model, qwha_indices  # noqa: E402
import qwha_hadamard as QH  # noqa: E402
from fourierft_fast import peft_indices  # noqa: E402

FAIL = []
def chk(i, name, ok, det=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {i}  {name}" + (f"   {det}" if det else ""))
    if not ok: FAIL.append(i)

print("=" * 78); print("  R.187 -- QWHA integration gates"); print("=" * 78)

# G0 -- vendored hadamard.py is the authors' file (modulo the declared guarded import)
theirs = open(os.path.join(ROOT, "temp/qwha/peft/src/peft/tuners/qwha/hadamard.py")).read()
ours = open(os.path.join(HERE, "qwha_hadamard.py")).read()
def norm(t):
    return "".join(t.split())[-4000:]          # tail, whitespace-insensitive
chk(0, "vendored hadamard.py matches the authors' (tail, ws-insensitive)",
    norm(theirs) == norm(ours), f"len ours {len(ours)}, theirs {len(theirs)}")

# G1 -- FORWARD bit-identical to the authors' layer
DRIVER = r'''
import sys, torch
sys.path.insert(0, "%s")
from peft.tuners.qwha.layer import QWHALinear as TheirQWHA
torch.manual_seed(0)
base = torch.nn.Linear(%d, %d, bias=False)
lay = TheirQWHA(base, "default", n_frequency=%d, scaling=%f, random_loc_seed=%d)
sd = torch.load("%s")
with torch.no_grad():
    lay.qwha_spectrum["default"].copy_(sd["spectrum"])
torch.save({"y": lay(sd["x"]).detach(), "idx": lay.qwha_indices["default"].detach()}, "%s")
'''
def bit_identity(in_f, out_f, k, scaling, seed):
    torch.manual_seed(0)
    base = torch.nn.Linear(in_f, out_f, bias=False)
    ours_layer = QWHALinear(base, n_frequency=k, scaling=scaling, random_loc_seed=seed)
    x = torch.randn(6, in_f)
    td = tempfile.mkdtemp(); fin, fout = os.path.join(td,"in.pt"), os.path.join(td,"out.pt")
    torch.save({"spectrum": ours_layer.spectrum.detach(), "x": x}, fin)
    drv = os.path.join(td, "d.py")
    open(drv,"w").write(DRIVER % (os.path.join(ROOT,"temp/qwha/peft/src"), in_f, out_f, k, scaling, seed, fin, fout))
    r = subprocess.run([sys.executable, drv], capture_output=True, text=True, cwd=ROOT)
    if r.returncode != 0:
        return None, (r.stderr.strip().splitlines() or ["subprocess failed"])[-1], None
    d = torch.load(fout)
    ours_y = ours_layer(x).detach()
    same_idx = torch.equal(ours_layer.indices, d["idx"].cpu())
    return torch.equal(ours_y, d["y"]), f"max|diff| = {(ours_y-d['y']).abs().max():.3e}", same_idx

for (m,n,k,sc) in ((768,768,256,150.0), (768,768,1000,150.0)):
    ok, det, same_idx = bit_identity(n, m, k, sc, 777)
    if ok is None: chk(1, f"forward bit-identical (d={m}, k={k})", False, det)
    else:
        chk(1, f"forward bit-identical to authors' layer (d={m}, k={k}, scaling={sc})", ok, det)
        chk(2, f"support draw identical to authors' (d={m}, k={k})", bool(same_idx))

# G3 -- the support draw IS FourierFT's [R.185 2]
ours_idx = qwha_indices(768, 768, 256, 777)
fft_idx = peft_indices(768, 768, 256, 777).long()
chk(3, "QWHA's support draw == PEFT FourierFT's randperm, verbatim",
    torch.equal(ours_idx, fft_idx), "same seeded randperm line")

# G4 -- the frozen path is an identity: spectrum=0 reproduces the base linear
L0 = QWHALinear(torch.nn.Linear(768,768,bias=False), n_frequency=256, init_weights=True)
x = torch.randn(8,768)
err = float((L0(x) - L0.base_layer(x)).abs().max())
chk(4, "spectrum=0 reproduces the frozen linear (wht(W)wht(x)/n == Wx)", err < 1e-5, f"max|diff| {err:.2e}")

# G5 -- wht/iwht round-trip on the torch path, at d=768 (K=12)
xd = torch.randn(4,768, dtype=torch.float64)
rt = float((QH.iwht(QH.wht(xd)) - xd).abs().max())
hadK, K = QH.get_hadK(768)
chk(5, "wht/iwht round-trip at d=768 on the pure-torch path", rt < 1e-5,
    f"max|err| {rt:.2e}, K={K} (768 = 12 x 64)")

# G6 -- wrapper adapts exactly the targets; base frozen, head trainable
class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Linear(768,768,bias=False)
        self.key = torch.nn.Linear(768,768,bias=False)
        self.value = torch.nn.Linear(768,768,bias=False)
        self.classifier = torch.nn.Linear(768,2)
    def forward(self,x): return self.classifier(self.value(self.key(self.query(x))))
t = Toy(); w = get_qwha_adapter_model(t, ["query","value"], n_frequency=256)
adapted = {n_ for n_,m_ in t.named_modules() if isinstance(m_, QWHALinear)}
chk(6, "wraps exactly query+value; base frozen, head trainable",
    adapted == {"query","value"}
    and all(not p.requires_grad for p in t.query.base_layer.parameters())
    and all(p.requires_grad for p in t.classifier.parameters()),
    f"adapted {sorted(adapted)}")

# G7 -- budget is k per module, like every other arm at matched budget
chk(7, "trainable params = k per adapted module",
    sum(p.numel() for p in t.query.parameters() if p.requires_grad) == 256, "256")

# G8 -- [R.308] the sync-free timing path is ARITHMETICALLY IDENTICAL.
# `torch.sparse_coo_tensor` on GPU indices forces 2 device syncs per forward --
# the [R.101]/[R.104] defect class, here in a BASELINE, where it would make the
# baseline look slow and flatter OURS.  `sync_free=True` must change the kernel
# schedule and NOTHING else: forward AND gradient bit-equal, or the repair is
# not a repair.  The vendored path stays the DEFAULT so prior results stand.
_b = torch.nn.Linear(768, 768)
_a0 = QWHALinear(_b, n_frequency=256, scaling=26.5165, random_loc_seed=777)
_a1 = QWHALinear(_b, n_frequency=256, scaling=26.5165, random_loc_seed=777,
                 sync_free=True)
_a1.spectrum.data.copy_(_a0.spectrum.data)
_x = torch.randn(64, 768)
_y0, _y1 = _a0(_x), _a1(_x)
_g0 = torch.autograd.grad(_y0.square().sum(), _a0.spectrum)[0]
_g1 = torch.autograd.grad(_y1.square().sum(), _a1.spectrum)[0]
chk(8, "[R.308] sync_free path is bit-identical in forward AND gradient",
    torch.equal(_y0, _y1) and torch.equal(_g0, _g1),
    f"fwd equal={torch.equal(_y0,_y1)} grad equal={torch.equal(_g0,_g1)}")
chk(8, "the vendored (syncing) path remains the DEFAULT",
    QWHALinear(_b, n_frequency=256).sync_free is False, "default sync_free=False")

# ⛔ AND ON CUDA.  The CPU check above passed while the CUDA forward differed in
# the last ulp (sqrt(n) was being computed on the device instead of moved from
# the host).  A repair gated only on CPU is not gated.
if torch.cuda.is_available():
    _bc = torch.nn.Linear(768, 768).cuda()
    _c0 = QWHALinear(_bc, n_frequency=256, scaling=26.5165, random_loc_seed=777).cuda()
    _c1 = QWHALinear(_bc, n_frequency=256, scaling=26.5165, random_loc_seed=777,
                     sync_free=True).cuda()
    _c1.spectrum.data.copy_(_c0.spectrum.data)
    _xc = torch.randn(256, 768, device="cuda")
    _z0, _z1 = _c0(_xc), _c1(_xc)
    _h0 = torch.autograd.grad(_z0.square().sum(), _c0.spectrum)[0]
    _h1 = torch.autograd.grad(_z1.square().sum(), _c1.spectrum)[0]
    chk(8, "[R.308] sync_free is bit-identical ON CUDA too (fwd AND grad)",
        torch.equal(_z0, _z1) and torch.equal(_h0, _h1),
        f"fwd equal={torch.equal(_z0,_z1)} grad equal={torch.equal(_h0,_h1)}")

print("=" * 78)
print(f"{9-len(set(FAIL))}/9 gate groups pass" if not FAIL else f"FAILED: {sorted(set(FAIL))}")
print("=" * 78)
sys.exit(1 if FAIL else 0)
