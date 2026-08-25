"""Blocked Walsh-Hadamard sparse adapter  (Phase J.10 -- the R2/R3 arm).

    dW = A_m^T C A_n        A = (I_{d/B} (x) H_B) . P     [blocked WHT + a fixed
                            C = k-sparse learned core      permutation]

--------------------------------------------------------------------------
WHERE THIS COMES FROM
--------------------------------------------------------------------------
J.9 (`llmdocs/J9_q1.md`) derived and verified the **coverage law**

    PR/d^2  =  lambda / (3 (1 + lambda)),   lambda = C_cells / (d^2 <1/(p_u p_v)>)

(verified to <= 2.4% over five decades of lambda, calibrated with NO fitting on
the Haar and FourierFT endpoints).  It says the normalised participation ratio
of dW is a **pure function of the row participation ratio `p` of the transform
`A`** -- not of the transform's identity, not of its conditioning, not of the
support pattern.  A blocked WHT has row PR **exactly B**, so the required block
size is a finite constant `B ~ 10 d / sqrt(mu k)`.

The J.7 Haar arm sits at `PR/d^2 = 0.00993` (34x below the FourierFT bar) and
paid 1.8-2.9% of accuracy for it.  This arm buys the delocalisation back inside
the orthogonal class, for 3-5 extra additions per element.

--------------------------------------------------------------------------
CONSTRUCTION
--------------------------------------------------------------------------
1.  TRANSFORM.  `A = D . Bl . P` with

    P  -- a fixed permutation of the `d` coordinates (0 flops: pure gather),
          drawn once from a seed derived deterministically from the support
          seed.  It decides which coordinates share a block; it does NOT change
          any row statistic (every row of `A` has support exactly `B` whatever
          `P` is), so nothing about it is tunable.
    Bl -- block-diagonal, block `i` = the UNNORMALISED Sylvester Hadamard
          `H_{B_i}` (entries +-1).  Applied by the in-place fast WHT butterfly:
          `log2(B)` stages, `B` additions per stage per block
          => **`d log2 B` ADDITIONS and ZERO MULTIPLICATIONS**.
    D  -- 1/sqrt(B_i) per coordinate.  When every block has the same size (the
          operating case: `d = 768, B = 256` -> exactly 3 blocks) this is a
          SINGLE GLOBAL SCALAR, folded into the a-priori output scale, so the
          forward performs **literally zero multiplications** in the transform.

    A A^T = D Bl P P^T Bl^T D = D (B I) D = I  =>  A is exactly orthogonal.
    (`H_B` Sylvester is symmetric, so the same routine applies `Bl` and `Bl^T`.)

    [derived] cost(A_d) = d * log2(B) additions,  0 mults.   NO log(d) FACTOR:
    at FIXED `B` the count doubles exactly when `d` doubles.

2.  BLOCK SIZE  B = 256  -- FIXED A PRIORI, ACCURACY-INDEPENDENT.
    J.9's coverage law requires `B ~ 10 d / sqrt(mu k)` to hold `PR/d^2 ~ 1/3`;
    at `d = 768, mu = 4, k = 1000` that is `B ~ 121`.  256 is the next power of
    two, with margin.  Measured (J.9 8.2): `PR/d^2 = 0.33268` = **99.5%** of the
    FourierFT bar.  NOT swept, and not revisited after any accuracy number.

3.  PLACEMENT MULTIPLICITY  mu = 4  -- FIXED A PRIORI ON THE RANK ARGUMENT.
    `A` orthogonal => `sv(A_m^T C A_n) = sv(C)`, so every rank statistic is a
    property of the sparse core alone and is set by the number of occupied
    cells `mu k`.  Measured (J.9 R7, transform-independent), stable rank vs the
    FourierFT bar 101.08:

        mu = 1 ->  66.71  (0.660x)      mu = 2 ->  90.46  (0.895x)
        mu = 4 -> 117.80  (1.165x)      mu = 8 -> 146.37  (1.448x)

    R1(b) demands *matching or exceeding* the bar: mu = 4 is the SMALLEST
    multiplicity that clears it.  It costs ZERO extra parameters (still `k` real
    scalars per module) but it is NOT free -- the core term is `4 mu k` flops
    per token, and `mu k = 4000 > m + n = 1536`, so mu is a genuine cost/rank
    knob.  mu = 8 is rejected on that cost, mu = 2 on the rank bar.  Both
    criteria are accuracy-independent and were settled before training.

4.  NORMALISATION -- DERIVED A PRIORI, NOTHING SWEPT (the J.6 lesson).
    Under AdamW the step is ~lr per coefficient regardless of gradient scale, so
    the per-parameter ATOM Frobenius norm ||d dW / d theta_j||_F IS the effective
    learning rate on dW.  A 7.24x mismatch once cost this program two phases.

        A orthogonal => ||A_m^T X A_n||_F = ||X||_F  =>  atom_bwht = s * sqrt(mu)
        FourierFT    => atom_fft = scaling / sqrt(2 m n)

        =>   s = fourierft_scaling / sqrt(2 * mu * m * n)          [derived]

    which is J.7's rule VERBATIM (same formula, same constant, only mu differs).
    Note `s sqrt(mu) = scaling / sqrt(2 m n)` is INDEPENDENT of mu, so the atom
    norm is 0.138106793200498 at `d = 768, scaling = 150` for every mu -- the
    same number the FourierFT and Haar arms carry.  Spread across coefficients
    is exactly zero: each parameter owns exactly mu unit cells and `A` is
    orthogonal on both sides, so ||atom||_F = s sqrt(mu) identically.
    init_std = 1.0 is PEFT's own `torch.randn(n_frequency)` verbatim.

5.  FORWARD  dW x = A_m^T( C ( A_n x ) ):  per token
        n log2 B (analysis, adds only) + 4 mu k (core) + m log2 B (synthesis)
        + m (residual add)
    => **Theta(b*(m + n + mu*k))** -- and the `mu*k` qualifier is MANDATORY, not
    decoration.  mu = 4 is what lifts stable rank to 117.80 (1.165x the FourierFT
    bar) and is the reason R1(b) is cleared, but the core term is `4*mu*k` flops
    per token, so at d = 768 `mu*k = 4000 > m + n = 1536` and **the core, not the
    transform, dominates** (16,000 of 29,056 flops/token = 55%).  As `d` grows at
    fixed `k` the `(m+n)` term overtakes it -- already at d = 4096,
    `m + n = 8192 > 4000` -- so the method is cleanly Theta(b(m+n)) asymptotically.
    `fourierft-fast` carries its own ~6k per-token term, so the comparison is fair;
    both numbers must always be reported.  NEVER state this arm's cost as
    `Theta(b(m+n) + k)` without the `mu*k` qualifier.
    Deterministic and EXACT -- no stochastic forward,
    no dense m x n tensor, identical path in train and eval.

--------------------------------------------------------------------------
HONEST ASYMPTOTIC CAVEAT (carried from J.9 4 -- must never be dropped)
--------------------------------------------------------------------------
At FIXED `B` the algorithm is exactly Theta(d): the counted op-count is
`d log2 B`, whose ratio on doubling `d` is exactly 2.0000, with no log factor.
R1(a) is an *algorithmic* op-count requirement and is met faithfully at any
fixed `B`.  BUT `lambda = mu k B^2 / d^2`, so holding `PR/d^2 ~ 1/3` as
`d -> infinity` AT FIXED `k` would require `B ~ d / sqrt(mu k)`, i.e.
`log2 B = log2 d - (1/2) log2(mu k)` stages, i.e. Theta(d log d).
**The stronger claim -- that PR/d^2 is MAINTAINED for all d at fixed k in
Theta(d) -- is false and must never be made.**  (FourierFT itself pays
Theta(d log d) for exactly this property.)

--------------------------------------------------------------------------
NOVELTY (R3) -- see `llmdocs/J10_bwht.md` 5 for the prior-art clearance
--------------------------------------------------------------------------
Blocked / lapped orthogonal transforms are OLD (JPEG's 8x8 block DCT; the
permuted variant is the SRHT/FJLT skeleton).  This module does not claim the
transform as novel.  The R3 case rests on (i) the coverage law, which is what
tells you `B` is a finite constant, and (ii) the decisive ablation against the
J.7 Haar arm at matched `k`, matched Theta(d) cost class and matched atom norm.
"""
from __future__ import annotations

import contextlib
import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
#  The blocked Walsh-Hadamard transform                                        #
# --------------------------------------------------------------------------- #

def bwht_block_sizes(d: int, B: int) -> List[int]:
    """Decompose `d` into blocks of power-of-two size, none larger than `B`,
    largest first.  `d = 768, B = 256` -> [256, 256, 256] (the operating case).
    A remainder is split into descending powers of two so the construction is
    defined -- and exactly orthogonal -- at every dimension."""
    if B < 1 or (B & (B - 1)):
        raise ValueError(f"B must be a power of two, got {B}")
    sizes: List[int] = []
    rem, blk = d, B
    while rem > 0:
        if blk <= rem:
            cnt = rem // blk
            sizes.extend([blk] * cnt)
            rem -= cnt * blk
        blk //= 2
        if blk == 0 and rem > 0:                                # unreachable
            raise ValueError(f"cannot tile d={d} with powers of two <= {B}")
    return sizes


def bwht_runs(sizes: Sequence[int]) -> List[Tuple[int, int]]:
    """[(block_size, count), ...] for consecutive equal sizes."""
    runs: List[Tuple[int, int]] = []
    for s in sizes:
        if runs and runs[-1][0] == s:
            runs[-1] = (s, runs[-1][1] + 1)
        else:
            runs.append((s, 1))
    return runs


def fwht_unnorm(x: torch.Tensor, B: int) -> torch.Tensor:
    """Unnormalised fast Walsh-Hadamard on the last axis of `x: (..., nb, B)`.

    `log2(B)` butterfly stages; each stage emits `B/2` sums and `B/2`
    differences per block => `B` ADDITIONS per block per stage, and **no
    multiplications at all**.  Total `B log2 B` additions per block.
    """
    lead = tuple(x.shape[:-1])
    h = 1
    while h < B:
        v = x.reshape(*lead, B // (2 * h), 2, h)
        a, b = v[..., 0, :], v[..., 1, :]
        x = torch.stack((a + b, a - b), dim=-2).reshape(*lead, B)
        h *= 2
    return x


def block_wht_unnorm(x: torch.Tensor, runs: Sequence[Tuple[int, int]]) -> torch.Tensor:
    """`Bl x` -- block-diagonal unnormalised Hadamard.  x: (..., d)."""
    if len(runs) == 1:
        B, cnt = runs[0]
        lead = tuple(x.shape[:-1])
        return fwht_unnorm(x.reshape(*lead, cnt, B), B).reshape(*lead, B * cnt)
    outs, off = [], 0
    lead = tuple(x.shape[:-1])
    for B, cnt in runs:
        seg = x[..., off:off + B * cnt].reshape(*lead, cnt, B)
        outs.append(fwht_unnorm(seg, B).reshape(*lead, B * cnt))
        off += B * cnt
    return torch.cat(outs, dim=-1)


@contextlib.contextmanager
def _exact_matmul(hmats: Optional[dict]):
    """Force the adapter's block GEMMs to true fp32.

    TF32 truncates the mantissa to 10 bits, which would make the GEMM
    realisation differ from the butterfly at ~1e-3 relative and break P.1's
    claim that the two realise the SAME `dW`.  This is a CPU-side flag, so it
    costs no kernel launch.  It must wrap the backward too -- forward and
    backward at different precisions is a silent gradient bug.
    """
    if hmats is None or hmats.get("_tf32", False) or not torch.cuda.is_available():
        yield
        return
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


def hadamard_dense(B: int, dtype=torch.float64) -> torch.Tensor:
    """The dense unnormalised Hadamard matrix `H_B` (entries +/-1), built by
    running `fwht_unnorm` on the identity so it is EXACTLY the same linear map
    the butterfly realises -- never an independently-derived matrix.

    `H_B` is symmetric, so `Bl x` for one block is the single GEMM `x @ H_B`.
    """
    eye = torch.eye(B, dtype=dtype)
    return fwht_unnorm(eye, B)


def block_wht_gemm(x: torch.Tensor, runs: Sequence[Tuple[int, int]],
                   hmats: dict) -> torch.Tensor:
    """`Bl x` -- identical map to `block_wht_unnorm`, realised as ONE batched
    GEMM per distinct block size instead of `log2(B)` sequential butterfly
    stages.

    P.1's measured point: on this hardware the adapter is dispatch-bound, so
    `B` multiply-accumulates per element in one kernel beats `log2(B)` additions
    per element spread over `log2(B)` dependent kernels.  Arithmetically this is
    STRICTLY WORSE (B vs log2 B per element) and that is reported, not hidden.
    """
    if len(runs) == 1:
        B, cnt = runs[0]
        lead = tuple(x.shape[:-1])
        return torch.matmul(x.reshape(*lead, cnt, B),
                            hmats[B]).reshape(*lead, B * cnt)
    outs, off = [], 0
    lead = tuple(x.shape[:-1])
    for B, cnt in runs:
        seg = x[..., off:off + B * cnt].reshape(*lead, cnt, B)
        outs.append(torch.matmul(seg, hmats[B]).reshape(*lead, B * cnt))
        off += B * cnt
    return torch.cat(outs, dim=-1)


def bwht_norm_vector(sizes: Sequence[int], dtype=torch.float64) -> torch.Tensor:
    """The diagonal `D`: 1/sqrt(B_i) on the coordinates of block `i`."""
    return torch.cat([torch.full((B,), B ** -0.5, dtype=torch.float64)
                      for B in sizes]).to(dtype)


def bwht_perm_seed(support_seed: int, d: int) -> int:
    """Deterministic, a priori: the permutation seed is a pure function of the
    support seed and the dimension.  Never chosen, never swept."""
    return int((support_seed * 65537 + d) % (2 ** 31 - 1))


def bwht_perm(d: int, support_seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(bwht_perm_seed(support_seed, d))
    return torch.randperm(d, generator=g)


def bwht_matrix(d: int, B: int, support_seed: int = 777,
                dtype=torch.float64) -> torch.Tensor:
    """Dense `A` (d x d) -- REFERENCE / MEASUREMENT ONLY, never in the forward."""
    sizes = bwht_block_sizes(d, B)
    perm = bwht_perm(d, support_seed)
    eye = torch.eye(d, dtype=dtype)
    # row j of `cols` is A e_j = column j of A  =>  A = cols^T
    cols = block_wht_unnorm(eye.index_select(-1, perm), bwht_runs(sizes)) \
        * bwht_norm_vector(sizes, dtype)
    return cols.T.contiguous()


def bwht_flops(d: int, B: int) -> dict:
    """[derived] Exact real-flop count for one length-d transform."""
    sizes = bwht_block_sizes(d, B)
    adds = sum(int(b * math.log2(b)) for b in sizes)
    uniform = len(set(sizes)) == 1
    return dict(adds=adds, mults=0 if uniform else d, total=adds + (0 if uniform else d),
                blocks=len(sizes), sizes=sizes, uniform=uniform,
                stages=int(math.log2(sizes[0])) if uniform else None)


# --------------------------------------------------------------------------- #
#  Support                                                                     #
# --------------------------------------------------------------------------- #

def bwht_support(m: int, n: int, k: int, mu: int, seed: int):
    """Return (rows, cols, pidx) with `mu * k` distinct cells; parameter j owns
    cells perm[j], perm[j + k], ..., perm[j + (mu-1)k].

    Mirrors PEFT FourierFT's own draw exactly -- and J.7's Haar arm verbatim:
    `torch.randperm(m*n, generator=manual_seed(seed))[:mu*k]`.  For mu = 1 the
    support IS the FourierFT arm's support; at any mu its first k cells are.
    """
    if mu * k > m * n:
        raise ValueError(f"mu*k = {mu*k} exceeds m*n = {m*n}")
    flat = torch.randperm(m * n,
                          generator=torch.Generator().manual_seed(seed))[:mu * k]
    rows, cols = flat // n, flat % n
    pidx = torch.arange(k).repeat(mu)
    return rows, cols, pidx


# --------------------------------------------------------------------------- #
#  K.2 -- the EQUITABLE (degree-balanced) support.  OPT-IN, default OFF.       #
#  `bwht_support` above is untouched; `support="random"` is bit-identical to   #
#  the shipped base.  See `llmdocs/K2_equitable.md`.                           #
# --------------------------------------------------------------------------- #

def _equitable_sequence(length: int, size: int, g: torch.Generator) -> torch.Tensor:
    """Length-`length` sequence over {0..size-1} whose value multiplicities all lie
    in {floor(length/size), ceil(length/size)} -- the EQUITABLE degree sequence --
    laid out as consecutive blocks of `size` that are each a random permutation
    (the last block a random partial one).  Distinct values inside every block."""
    out, rem = [], length
    while rem > 0:
        p = torch.randperm(size, generator=g)
        out.append(p[:min(size, rem)])
        rem -= min(size, rem)
    return torch.cat(out)


def bwht_support_equitable(m: int, n: int, k: int, mu: int, seed: int,
                           max_repair: int = 100000):
    """K.2: `mu*k` DISTINCT cells placed as a union of random (partial)
    permutation matchings, so that

      * every row degree is in {floor(mu k/m), ceil(mu k/m)}   (equitable),
      * every column degree is in {floor(mu k/n), ceil(mu k/n)} (equitable),
      * every parameter owns exactly `mu` cells, in `mu` DISTINCT rows,
        `mu` DISTINCT columns and `mu` DISTINCT matchings.

    Cell COUNT is identical to `bwht_support` -- so the forward op-count, the
    memory transient and the atom norm `s*sqrt(mu)` are untouched by construction.
    Only the placement changes.  Delta = ceil(mu k / min(m,n)) matchings; a
    Delta-regular bipartite graph decomposes into Delta matchings (Koenig), and
    matching 0 is a FULL permutation, so rank(C) = min(m,n) exactly at every mu.
    """
    ncell = mu * k
    if ncell > m * n:
        raise ValueError(f"mu*k = {ncell} exceeds m*n = {m * n}")
    g = torch.Generator().manual_seed(seed)

    rows = _equitable_sequence(ncell, m, g)
    cols = _equitable_sequence(ncell, n, g)

    # --- chunks: maximal runs over which BOTH rows and cols are distinct ----- #
    bound = sorted({0, ncell}
                   | {i for i in range(m, ncell, m)}
                   | {i for i in range(n, ncell, n)})
    chunks = [(bound[i], bound[i + 1]) for i in range(len(bound) - 1)]
    cid = torch.empty(ncell, dtype=torch.long)
    for c, (a, b) in enumerate(chunks):
        cid[a:b] = c

    def _rint(hi):
        return int(torch.randint(hi, (1,), generator=g).item())

    # --- repair 1: distinct CELLS.  Swap columns inside one chunk (preserves  #
    #     the matching property and every row/column degree exactly).          #
    used = {}
    for q in range(ncell):
        used.setdefault((int(rows[q]), int(cols[q])), []).append(q)
    dup = [qs[i] for qs in used.values() if len(qs) > 1 for i in range(1, len(qs))]
    live = {kk for kk, vv in used.items()}
    steps = 0
    while dup:
        q = dup.pop()
        a, b = chunks[int(cid[q])]
        ok = False
        while not ok:
            steps += 1
            if steps > max_repair:
                raise RuntimeError("equitable support: cell de-dup did not converge")
            q2 = a + _rint(b - a)
            if q2 == q:
                continue
            c1 = (int(rows[q]), int(cols[q2]))
            c2 = (int(rows[q2]), int(cols[q]))
            if c1 in live or c2 in live or c1 == c2:
                continue
            live.discard((int(rows[q]), int(cols[q])))
            live.discard((int(rows[q2]), int(cols[q2])))
            cols[q], cols[q2] = cols[q2].clone(), cols[q].clone()
            live.add(c1)
            live.add(c2)
            ok = True
        # recount duplicates lazily: rebuild once the queue empties
        if not dup:
            used = {}
            for t in range(ncell):
                used.setdefault((int(rows[t]), int(cols[t])), []).append(t)
            dup = [qs[i] for qs in used.values() if len(qs) > 1
                   for i in range(1, len(qs))]

    # --- deal cells to parameters: cell q -> parameter q mod k.  Every chunk  #
    #     is contiguous and of length <= min(m,n), so if min(m,n) <= k each    #
    #     chunk hits `len(chunk)` DISTINCT parameters => mu distinct matchings.#
    pidx = torch.arange(ncell) % k

    # --- repair 2: per-parameter DISTINCT rows and DISTINCT columns.  Swap    #
    #     the cell<->parameter labels of two cells inside one chunk (changes   #
    #     neither the support nor any degree nor any parameter's cell count).  #
    def _bad(j_cells):
        r = [int(rows[t]) for t in j_cells]
        c = [int(cols[t]) for t in j_cells]
        return len(set(r)) != len(r) or len(set(c)) != len(c)

    rl, cl = rows.tolist(), cols.tolist()

    def _cost(qs):
        """0 iff the parameter's cells occupy |qs| distinct rows AND columns."""
        return (len(qs) - len(set(rl[t] for t in qs))) \
             + (len(qs) - len(set(cl[t] for t in qs)))

    owner = {}
    for q in range(ncell):
        owner.setdefault(int(pidx[q]), []).append(q)
    # Strict descent on the non-negative integer potential sum(_cost): every
    # accepted swap lowers it by >= 1, so the loop provably terminates.
    steps = 0
    while True:
        bad = [j for j, qs in owner.items() if _cost(qs) > 0]
        if not bad:
            break
        j = bad[_rint(len(bad))]
        c0 = _cost(owner[j])
        moved = False
        order = owner[j][:]
        for q in [order[i] for i in torch.randperm(len(order), generator=g).tolist()]:
            a, b = chunks[int(cid[q])]
            for q2 in range(a, b):
                steps += 1
                if steps > max_repair:
                    raise RuntimeError("equitable support: parameter repair "
                                       "did not converge")
                j2 = int(pidx[q2])
                if j2 == j:
                    continue
                n1 = [t for t in owner[j] if t != q] + [q2]
                n2 = [t for t in owner[j2] if t != q2] + [q]
                if _cost(n1) + _cost(n2) < c0 + _cost(owner[j2]):
                    pidx[q], pidx[q2] = pidx[q2].clone(), pidx[q].clone()
                    owner[j], owner[j2] = n1, n2
                    moved = True
                    break
            if moved:
                break
        if not moved:
            raise RuntimeError("equitable support: parameter repair stalled "
                               f"(no improving swap for parameter {j})")

    # --- final random relabelling of the k parameters (no structural effect) - #
    relabel = torch.randperm(k, generator=g)
    pidx = relabel[pidx]
    return rows.long(), cols.long(), pidx.long()


SUPPORTS = {"random": bwht_support, "equitable": bwht_support_equitable}


# --------------------------------------------------------------------------- #
#  The forward (never materialises dW)                                         #
# --------------------------------------------------------------------------- #

def bwht_delta_apply(x: torch.Tensor, vals: torch.Tensor,
                     rows: torch.Tensor, cols: torch.Tensor,
                     pidx: torch.Tensor,
                     runs_n, runs_m,
                     perm_n: torch.Tensor, invperm_m: torch.Tensor,
                     dn: Optional[torch.Tensor], dm: Optional[torch.Tensor],
                     m: int, hmats: Optional[dict] = None) -> torch.Tensor:
    """dW x = A_m^T ( C ( A_n x ) ),  x: (b, n) -> (b, m).  No m x n tensor.

    `A_n = D_n Bl_n P_n`  and  `A_m^T = P_m^T Bl_m D_m`  (Bl symmetric).
    When the blocks are uniform, `dn`/`dm` are None because the two scalar
    normalisations have been folded into `vals` -- so the transform performs
    ZERO multiplications.
    """
    # `hmats is None` -> the shipped butterfly path, bit-identical to pre-P.1.
    if hmats is None:
        _bl = block_wht_unnorm
    else:
        def _bl(t, runs):
            return block_wht_gemm(t, runs, hmats)
    with _exact_matmul(hmats):
        z = _bl(x.index_select(-1, perm_n), runs_n)             # (b, n) Bl_n P_n x
        if dn is not None:
            z = z * dn
        contrib = z.index_select(-1, cols) * vals[pidx]         # (b, mu*k)
        y = torch.zeros(x.shape[0], m, dtype=z.dtype, device=z.device)
        y = y.index_add(-1, rows, contrib)                      # (b, m)  C z
        if dm is not None:
            y = y * dm
        return _bl(y, runs_m).index_select(-1, invperm_m)       # P_m^T Bl_m .


class _BwhtFn(torch.autograd.Function):
    """Recomputation variant: MARGINAL stash is O(k), with no term in `b`.

    The naive graph would stash the gathered `z[:, cols]`, a Theta(b * mu * k)
    tensor, to form the gradient w.r.t. the coefficients.  Here the forward runs
    under `no_grad` and the backward rebuilds the chain from `x`, which the
    frozen base `nn.Linear` is already holding -- so the adapter adds no
    per-token bytes at all (R1a's stash clause).  Cost: one extra adapter
    forward, i.e. ~(m+n) log2 B + 4 mu k flops per token in the backward.
    """

    @staticmethod
    def forward(ctx, x, vals, rows, cols, pidx, runs_n, runs_m,
                perm_n, invperm_m, dn, dm, m, hmats=None):
        with torch.no_grad():
            out = bwht_delta_apply(x, vals, rows, cols, pidx, runs_n, runs_m,
                                   perm_n, invperm_m, dn, dm, m, hmats)
        ctx.save_for_backward(x, vals)
        ctx.tables = (rows, cols, pidx, runs_n, runs_m, perm_n, invperm_m,
                      dn, dm, m, hmats)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, vals = ctx.saved_tensors
        (rows, cols, pidx, runs_n, runs_m, perm_n, invperm_m,
         dn, dm, m, hmats) = ctx.tables
        need_x = ctx.needs_input_grad[0]
        with _exact_matmul(hmats), torch.enable_grad():
            xd = x.detach().requires_grad_(need_x)
            vd = vals.detach().requires_grad_(True)
            out = bwht_delta_apply(xd, vd, rows, cols, pidx, runs_n, runs_m,
                                   perm_n, invperm_m, dn, dm, m, hmats)
            tgt = [xd, vd] if need_x else [vd]
            grads = torch.autograd.grad(out, tgt, grad_out.contiguous(),
                                        allow_unused=True)
        gx = grads[0] if need_x else None
        gv = grads[1] if need_x else grads[0]
        return (gx, gv) + (None,) * 11


class BwhtLinear(nn.Module):
    """Frozen base `nn.Linear` + exact blocked-WHT-domain sparse adapter."""

    def __init__(self, base_layer: nn.Linear, n_frequency: int = 1000,
                 block: int = 256, mu: int = 4, support_seed: int = 777,
                 fourierft_scaling: float = 150.0,
                 scaling: Optional[float] = None,
                 init_std: float = 1.0, no_recompute: bool = False,
                 support: str = "random",
                 realization: str = "butterfly", gemm_tf32: bool = False):
        super().__init__()
        # P.1: `realization` changes ONLY how the fixed orthogonal map `A` is
        # evaluated -- never `A` itself, never `dW`, never the support, never
        # the atom norm.  "butterfly" is the shipped default, so every existing
        # call is bit-identical to pre-P.1.
        if realization not in ("butterfly", "gemm"):
            raise ValueError(f"realization must be butterfly|gemm, got {realization!r}")
        self.realization = realization
        # TF32 truncates the mantissa to 10 bits; a B=256 GEMM against a +/-1
        # Hadamard would then differ from the butterfly at ~1e-3 relative, which
        # fails P.1's G1 gate.  Exact by default; TF32 is an explicit ablation.
        self.gemm_tf32 = bool(gemm_tf32)
        # `no_recompute=True` keeps the naive autograd graph (stashes the
        # Theta(b*mu*k) gather).  ABLATION / gate use only -- the default path
        # recomputes and stashes nothing marginal.
        self.no_recompute = bool(no_recompute)
        self.base_layer = base_layer
        for p in self.base_layer.parameters():
            p.requires_grad_(False)
        m, n = base_layer.out_features, base_layer.in_features
        self.m, self.n, self.mu, self.k, self.block = m, n, mu, n_frequency, block
        self.support_seed = support_seed
        self.fourierft_scaling = fourierft_scaling
        # --- the a-priori normalisation constant (see module docstring 4) ---
        if scaling is None:
            scaling = fourierft_scaling / math.sqrt(2.0 * mu * m * n)
        self.scaling = float(scaling)
        self.init_std = float(init_std)

        # K.2: opt-in placement.  `support="random"` is the SHIPPED base and
        # calls `bwht_support` unchanged -> bit-identical to pre-K.2.
        if support not in SUPPORTS:
            raise ValueError(f"support must be one of {sorted(SUPPORTS)}, got {support!r}")
        self.support = support
        rows, cols, pidx = SUPPORTS[support](m, n, n_frequency, mu, support_seed)
        self.register_buffer("rows", rows, persistent=False)
        self.register_buffer("cols", cols, persistent=False)
        self.register_buffer("pidx", pidx, persistent=False)

        self.sizes_n = bwht_block_sizes(n, block)
        self.sizes_m = bwht_block_sizes(m, block)
        self.runs_n = bwht_runs(self.sizes_n)
        self.runs_m = bwht_runs(self.sizes_m)
        self.uniform = (len(set(self.sizes_n)) == 1
                        and len(set(self.sizes_m)) == 1)

        perm_n = bwht_perm(n, support_seed)
        perm_m = bwht_perm(m, support_seed)
        self.register_buffer("perm_n", perm_n, persistent=False)
        self.register_buffer("invperm_m", torch.argsort(perm_m), persistent=False)
        self.register_buffer("perm_m", perm_m, persistent=False)

        wdt = base_layer.weight.dtype
        if self.uniform:
            # D_n and D_m are single global scalars -> fold them into the
            # coefficient scale so the transform does ZERO multiplications.
            self.fold = float(self.sizes_n[0] ** -0.5 * self.sizes_m[0] ** -0.5)
            self.dn = None
            self.dm = None
        else:
            self.fold = 1.0
            # built in fp64 then cast ONCE, so an fp64 layer gets exact 1/sqrt(B)
            self.register_buffer("dn_vec", bwht_norm_vector(self.sizes_n).to(wdt),
                                 persistent=False)
            self.register_buffer("dm_vec", bwht_norm_vector(self.sizes_m).to(wdt),
                                 persistent=False)
            self.dn = self.dn_vec
            self.dm = self.dm_vec

        # P.1: dense +/-1 Hadamard per distinct block size, built by running the
        # butterfly on the identity (so it is the SAME map, not a re-derivation)
        # in fp64 and cast once.  Buffers, so `.to(device)` carries them.
        self._hkeys = sorted({*self.sizes_n, *self.sizes_m}) if realization == "gemm" else []
        for B in self._hkeys:
            self.register_buffer(f"H_{B}", hadamard_dense(B).to(wdt), persistent=False)

        # PEFT's own init verbatim: `torch.randn(n_frequency)`, std = 1.0.
        self.spectrum = nn.Parameter(torch.randn(n_frequency, dtype=wdt) * init_std)

    # -- forward ----------------------------------------------------------- #
    def _tables(self):
        dn = getattr(self, "dn_vec", None) if not self.uniform else None
        dm = getattr(self, "dm_vec", None) if not self.uniform else None
        return dn, dm

    def _hmats(self):
        if self.realization != "gemm":
            return None
        d = {B: getattr(self, f"H_{B}") for B in self._hkeys}
        d["_tf32"] = self.gemm_tf32
        return d

    def delta_apply(self, x: torch.Tensor) -> torch.Tensor:
        shp = x.shape
        xf = x.reshape(-1, shp[-1])
        dn, dm = self._tables()
        vals = self.spectrum * (self.scaling * self.fold)
        fn = bwht_delta_apply if self.no_recompute else _BwhtFn.apply
        hmats = self._hmats()
        out = fn(xf, vals, self.rows, self.cols, self.pidx,
                 self.runs_n, self.runs_m, self.perm_n, self.invperm_m,
                 dn, dm, self.m, hmats)
        return out.reshape(*shp[:-1], self.m)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # EXACT and DETERMINISTIC; identical in train and eval.
        return self.base_layer(x) + self.delta_apply(x)

    # -- reference objects (measurement / merging only) -------------------- #
    @torch.no_grad()
    def get_delta_weight(self, dtype=torch.float64) -> torch.Tensor:
        """dW = A_m^T C A_n.  Materialises m x n -- NEVER called in the forward."""
        dev = self.spectrum.device
        C = torch.zeros(self.m, self.n, dtype=dtype, device=dev)
        C = C.index_put((self.rows, self.cols),
                        (self.spectrum.to(dtype) * self.scaling)[self.pidx])
        Am = bwht_matrix(self.m, self.block, self.support_seed, dtype).to(dev)
        An = bwht_matrix(self.n, self.block, self.support_seed, dtype).to(dev)
        return Am.T @ C @ An

    @torch.no_grad()
    def atom_frobenius(self, j: int = 0, dtype=torch.float64) -> float:
        """||d dW / d theta_j||_F -- the effective learning rate on dW (J.6)."""
        dev = self.spectrum.device
        C = torch.zeros(self.m, self.n, dtype=dtype, device=dev)
        sel = (self.pidx == j)
        C = C.index_put((self.rows[sel], self.cols[sel]),
                        torch.full((int(sel.sum()),), self.scaling, dtype=dtype,
                                   device=dev))
        # A orthogonal => ||A^T C A||_F = ||C||_F exactly; computed explicitly
        # anyway so the gate MEASURES rather than assumes it.
        Am = bwht_matrix(self.m, self.block, self.support_seed, dtype).to(dev)
        An = bwht_matrix(self.n, self.block, self.support_seed, dtype).to(dev)
        return float((Am.T @ C @ An).norm())

    def extra_repr(self) -> str:
        return (f"m={self.m}, n={self.n}, k={self.k}, B={self.block}, "
                f"mu={self.mu}, support={self.support}, cells={self.rows.numel()}, "
                f"blocks={self.sizes_m}/{self.sizes_n}, "
                f"scaling={self.scaling:.6g}, fold={self.fold:.6g}, "
                f"init_std={self.init_std:.4g}, uniform={self.uniform}")


# --------------------------------------------------------------------------- #
#  Op counts (exact, with constants)                                           #
# --------------------------------------------------------------------------- #

def flops_forward(m: int, n: int, k: int, mu: int = 4, block: int = 256) -> dict:
    """[derived] Real flops per TOKEN per module for the unmerged forward."""
    fn, fm = bwht_flops(n, block), bwht_flops(m, block)
    d = dict(
        analysis=float(fn["total"]),        # n log2 B additions
        core=2.0 * mu * k,                  # mu*k cells: one mult + one add = 2 flops/cell
        synthesis=float(fm["total"]),       # m log2 B additions
        residual_add=float(m),
    )
    d["total"] = sum(d.values())
    return d
    # ⚠️ ACCOUNTING NOTE (K_accounting_audit.md §5; applied 2026-07-23). `core` was
    # `4.0*mu*k`, double-counting each cell as 4 flops while the program's own
    # `bench_adapter_cost.theoretical_flops::sparseft_ideal` and the transform line
    # charge 2/cell (one mul + one add). Corrected to a single declared convention —
    # arithmetic-only, gathers free, residual add included for every arm. This changes
    # the reported flops/token (base 29,056 -> 21,056) but NO gate verdict, NO R1(a)
    # asymptotic (2*mu*k and 4*mu*k are both Theta(mu*k)), and NO rank/PR/atom-norm.


# --------------------------------------------------------------------------- #
#  Model wrapper                                                               #
# --------------------------------------------------------------------------- #

class BwhtAdapterModel(nn.Module):
    """Wrap a HF model, replacing target `nn.Linear`s with `BwhtLinear`.

    Mirrors `haar_adapter.HaarAdapterModel` so `train_glue.py` can drive it
    through the same `custom_methods` branch -- the two arms differ in exactly
    one thing, the transform.
    """

    def __init__(self, model: nn.Module, target_modules, n_frequency: int = 1000,
                 block: int = 256, mu: int = 4, seed: int = 777,
                 fourierft_scaling: float = 150.0,
                 scaling: Optional[float] = None, init_std: float = 1.0,
                 freeze_classifier_dense: bool = False,
                 support: str = "random", realization: str = "butterfly",
                 gemm_tf32: bool = False):
        super().__init__()
        self.model = model
        self.support = support
        self.realization = realization
        self.target_modules = list(target_modules)
        self.n_frequency, self.mu, self.seed, self.block = \
            n_frequency, mu, seed, block
        self.adapted_modules = []
        for p in model.parameters():
            p.requires_grad = False
        for name, module in list(model.named_modules()):
            if not isinstance(module, nn.Linear):
                continue
            if not any(t in name for t in self.target_modules):
                continue
            parts = name.rsplit(".", 1)
            if len(parts) == 2:
                parent = dict(model.named_modules())[parts[0]]
                attr = parts[1]
            else:
                parent, attr = model, parts[0]
            adapted = BwhtLinear(module, n_frequency=n_frequency, block=block,
                                 mu=mu, support_seed=seed,
                                 fourierft_scaling=fourierft_scaling,
                                 scaling=scaling, init_std=init_std,
                                 support=support, realization=realization,
                                 gemm_tf32=gemm_tf32)
            adapted.to(module.weight.device)
            setattr(parent, attr, adapted)
            self.adapted_modules.append(name)
        for name, p in model.named_parameters():
            if "classifier" in name or "score" in name:
                if freeze_classifier_dense and "classifier.dense" in name:
                    continue
                p.requires_grad = True

    def gradient_checkpointing_enable(self, **kw):
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable(**kw)

    def forward(self, **kw):
        return self.model(**kw)

    def print_trainable_parameters(self):
        tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        tot = sum(p.numel() for p in self.parameters())
        print(f"trainable params: {tr:,} || all params: {tot:,} || "
              f"trainable%: {tr / tot * 100:.4f}")
        return tr

    def get_adapter_params(self) -> int:
        return sum(p.numel() for n, p in self.named_parameters()
                   if p.requires_grad and "spectrum" in n)


def get_bwht_adapter_model(model: nn.Module, target_modules, **kw) -> BwhtAdapterModel:
    return BwhtAdapterModel(model, target_modules, **kw)
