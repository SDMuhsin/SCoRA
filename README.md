# LYRA: Low-Frequency Rank Adaptation via Factored DCT Coefficients

Code for the IEEE Signal Processing Letters paper *LYRA: Low-Frequency Rank Adaptation via Factored DCT
Coefficients for Parameter Efficient Fine Tuning of Transformers* (2026).
[doi:10.1109/LSP.2026.3714737](https://doi.org/10.1109/LSP.2026.3714737)

## Method

LYRA parameterizes each weight update in the 2D DCT-II domain using a small set of low-frequency
coefficients, chosen separately along each axis:

```
delta_W = gamma * C_m^T @ S @ C_n
```

where `C_m` and `C_n` are `p` and `q` selected rows of the frozen orthonormal DCT-II bases and `S` is the
trainable coefficient matrix. Because the selection is separable, the forward pass factors into three small
matrix multiplications and never materializes a dense `m x n` update:

```
y = Wx + gamma * ((x @ C_n^T) @ S^T) @ C_m
```

Cost is `O(b(nq + pq + pm))` instead of `O(bmn)`. Trainable parameters per module are `p*q` in dense mode, or
`p*r + r*q` when `S` is factored as `S = AB`. Both modes used in the paper hold 256 parameters per module
(dense `p=q=16`, factored `p=q=32, r=4`).

Note on naming: the method is called `spectral` throughout the code (`--optimizer adamw-spectral`,
`adapter_method=spectral` in the result CSVs).

## Setup

```bash
python -m venv env && source env/bin/activate
pip install -r requirements.txt
export HF_HOME=./data HF_DATASETS_CACHE=./data TORCH_HOME=./data PYTHONPATH=src
```

Use `--dtype float32`. Reduced precision causes significant loss in the DCT computations on these encoders.

## Reproducing the experiments

| What | Where |
|------|-------|
| Full GLUE and SuperGLUE sweep (all methods, both encoders) | `sbatch/run_peft_experiments.sh` |
| Single training run | `src/train_glue.py --optimizer adamw-spectral ...` |
| Spectral anatomy figure (Fig. 1) | `src/spectral_figure.py`, `--plot-only` reuses saved data |
| DCT energy check on full fine-tuning updates | `src/verify_dct_energy.py` |

`run_peft_experiments.sh` holds the per-task LYRA configurations and the baseline hyperparameters used in the
paper. `train_glue.py` runs seeds 41 to 45 and appends the median to `results/mo53_glue.csv`; all LYRA options
are the `--spectral_*` flags.

Published results are in `results/mo53_glue.csv` (BERT-base) and `results/mo53_glue_roberta.csv`
(RoBERTa-base).

## Repository layout

| Path | Contents |
|------|----------|
| `src/spectral_adapter.py` | LYRA implementation (`SpectralAdapterLinear`, `SpectralAdapterModel`) |
| `src/train_glue.py` | Training and evaluation harness for all methods |
| `src/spectral_figure.py` | Figure 1 |
| `src/dylora.py` | DyLoRA baseline (not available in the PEFT library) |
| `sbatch/` | Experiment launch scripts |
| `results/` | Result CSVs and figure data |
| `llmdocs/lyra/` | Paper sources and supporting notes |

`src/` also contains code from unrelated follow-up work; the files above are the ones used by the paper.

## Citation

```bibtex
@ARTICLE{11613143,
  author={Muhsin, Sayed and Ko, Seok-Bum},
  journal={IEEE Signal Processing Letters},
  title={LYRA: Low-Frequency Rank Adaptation via Factored DCT Coefficients for Parameter Efficient Fine Tuning of Transformers},
  year={2026},
  volume={33},
  pages={3073-3077},
  doi={10.1109/LSP.2026.3714737}}
```
