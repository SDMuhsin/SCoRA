# SCoRA: Sparse Cosine Rank Adaptation

Code for *SCoRA: Sparse Cosine Rank Adaptation for Parameter-Efficient Fine-Tuning of
Transformers*, under review at IEEE Signal Processing Letters.

## Method

SCoRA writes the weight update as a low-rank product and stores each of its two factors
as a sparse cosine spectrum:

```
delta_W = gamma * sum_j u_j v_j^T,    u_j = U beta_j,    v_j = V alpha_j
```

`U` and `V` are frozen row subsets of the orthonormal DCT-II matrix, drawn once from a
stored seed; `beta` and `alpha` are the only trainable tensors. The update is assembled
by an outer product of the two factors, so no inverse transform is taken. A module
trains `r(s+t)` coefficients, a count that depends on neither matrix dimension. The
scale `gamma` is fixed in closed form rather than searched.

## Source

| file | what it is |
|---|---|
| `src/slr_adapter.py` | the adapter: frames, initialization, the derived scale, both forward paths |
| `src/verify_slr.py` | its checks |
| `src/train_glue.py` | encoder training and evaluation |
| `src/train_clm.py` | causal-LM training, used by the rank-against-support ablation |
| `scripts/run_all_gates.py` | runs every instrument's checks |

Comparator implementations: `src/merged_fourierft.py` (FourierFT), `src/haar_adapter.py`
(WaveFT), `src/loca_adapter.py` (LoCA), `src/qwha_adapter.py` (QWHA),
`src/spectral_adapter.py` (LYRA).

## Running it

```
env/bin/python src/train_glue.py --task_name rte --model_name_or_path roberta-base \
  --optimizer adamw-slr --slr_rank 1 --slr_s 128 --slr_init zero --slr_seed 777 \
  --slr_target_modules query,value
```
