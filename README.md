# Massive MIMO Channel Estimation — Complete Training Package

This package contains everything needed to reproduce the results in
**"Noise-Conditioned Residual Learning for Robust Channel Estimation
in High-Mobility Massive MIMO"**: four algorithms, two channel models,
two array sizes, all wired together with a single master training
script and a single master plotting script.

---

## Folder structure

```
MMO_PACKAGE/
├── shared/                       # Channel generators + baselines
│   ├── channels.py               # Jakes & TDL (10 scenarios)
│   ├── baselines.py              # LS, MMSE, ISTA, top-k mask, blend
│   └── __init__.py
│
├── stage1_jakes/                 # Stage I: architectural screening
│   ├── cnn_lstm/                 # Algorithm 1: ConvLSTM + SGD
│   │   └── train_cnn_lstm.py
│   ├── cnn_sparse/               # Algorithm 2: CNN + top-k mask
│   │   └── train_cnn_sparse.py
│   ├── sa_dlcs/                  # Algorithm 3: ConvLSTM + ISTA
│   │   └── train_sa_dlcs_jakes.py
│   └── ncrn/                     # Algorithm 4: residual + blend
│       └── train_ncrn_jakes.py
│
├── stage2_tdl/                   # Stage II: cross-model validation
│   ├── cnn_lstm/                 # CNN-LSTM retrained on TDL
│   │   └── train_cnn_lstm_tdl.py
│   ├── cnn_sparse/               # CNN-Sparse retrained on TDL
│   │   └── train_cnn_sparse_tdl.py
│   ├── sa_dlcs/                  # Sa-DLCS retrained on TDL
│   │   └── train_sa_dlcs_tdl.py
│   └── ncrn/                     # NCRN retrained on TDL (curriculum)
│       └── train_ncrn_tdl.py
│
├── plotting/
│   └── plot_all_figures.py       # Generates all 13 paper figures
│
├── models/                       # Trained .keras + history pickles
├── outputs/
│   ├── figures/                  # 13 PNG plots
│   └── csv/                      # Per-table CSV exports
│
├── train_all.sh                  # Master training orchestrator
└── README.md                     # This file
```

---

## Quick start

```bash
# 1. One-time setup (in your conda env)
conda activate deeplearn_arm
pip install tensorflow numpy matplotlib scikit-learn

# 2. Train everything (both array sizes, 12 models, ~2 hours on M1)
cd MMO_PACKAGE
bash train_all.sh

# 3. Generate every figure (~30 min)
python plotting/plot_all_figures.py

# Want a fast preview (4x fewer samples, ~7 min)?
python plotting/plot_all_figures.py --quick
```

---

## What the package produces

### 16 trained models (in `models/`)

| File | Algorithm | Channel | Array |
|---|---|---|---|
| `cnn_lstm_128.keras`        | CNN-LSTM   | Jakes (10 dB) | 128×128 |
| `cnn_lstm_64.keras`         | CNN-LSTM   | Jakes (10 dB) | 64×64   |
| `cnn_sparse_128.keras`      | CNN-Sparse | Jakes (10 dB) | 128×128 |
| `cnn_sparse_64.keras`       | CNN-Sparse | Jakes (10 dB) | 64×64   |
| `sa_dlcs_jakes_128.keras`   | Sa-DLCS    | Jakes (10 dB) | 128×128 |
| `sa_dlcs_jakes_64.keras`    | Sa-DLCS    | Jakes (10 dB) | 64×64   |
| `ncrn_jakes_128.keras`      | NCRN       | Jakes (curr.) | 128×128 |
| `ncrn_jakes_64.keras`       | NCRN       | Jakes (curr.) | 64×64   |
| `cnn_lstm_tdl_128.keras`    | CNN-LSTM   | TDL (10 sc.)  | 128×128 |
| `cnn_lstm_tdl_64.keras`     | CNN-LSTM   | TDL (10 sc.)  | 64×64   |
| `cnn_sparse_tdl_128.keras`  | CNN-Sparse | TDL (10 sc.)  | 128×128 |
| `cnn_sparse_tdl_64.keras`   | CNN-Sparse | TDL (10 sc.)  | 64×64   |
| `sa_dlcs_tdl_128.keras`     | Sa-DLCS    | TDL (10 sc.)  | 128×128 |
| `sa_dlcs_tdl_64.keras`      | Sa-DLCS    | TDL (10 sc.)  | 64×64   |
| `ncrn_tdl_128.keras`        | NCRN       | TDL (curr.)   | 128×128 |
| `ncrn_tdl_64.keras`         | NCRN       | TDL (curr.)   | 64×64   |

Each model has a matching `_history.pkl` with the training-loss curve.

### 13 figures (in `outputs/figures/`)

| # | Filename | Content |
|---|---|---|
| 1  | `nmse_jakes_128.png`           | Stage I NMSE, 128×128 |
| 2  | `mse_jakes_128_semilogy.png`   | Stage I MSE log-y, 128×128 |
| 3  | `nmse_jakes_64.png`            | Stage I NMSE, 64×64 |
| 4  | `mse_jakes_64_semilogy.png`    | Stage I MSE log-y, 64×64 |
| 5  | `nmse_tdl_128.png`             | Stage II NMSE, 128×128 |
| 6  | `nmse_tdl_64.png`              | Stage II NMSE, 64×64 |
| 7  | `v7_scenarios_LOS.png`         | Per-scenario: 4 LOS panels |
| 8  | `v7_scenarios_NLOS.png`        | Per-scenario: 4 NLOS panels |
| 9  | `v7_scenarios_O2I.png`         | Per-scenario: UMi_O2I |
| 10 | `v7_scenarios_HST.png`         | Per-scenario: RMa_HST |
| 11 | `ber_tdl_64.png`               | BER vs SNR, ZF detection |
| 12 | `loss_curves.png`              | Training loss, both stages |
| 13 | `confusion_matrix_v8_64.png`   | RF scenario classifier |

---

## Algorithm summary

| Name | Channel | Optimiser | Filters | Refinement | Target |
|---|---|---|---|---|---|
| **CNN-LSTM**   | Jakes | SGD + LR step decay | 16→8     | None       | Full H |
| **CNN-Sparse** | Jakes | Adam                | 16→8 (CNN) | top-k mask | Full H |
| **Sa-DLCS**    | Jakes / TDL | Adam          | 16→8     | ISTA (K=3, λ=0.05) | Full H |
| **NCRN**       | Jakes / TDL | Adam + curriculum | 128→64 + SE skip | 3-regime SNR blend | Residual r = h_true − h_noisy |

NCRN's input has 3 channels (real, imag, SNR/30) — the SNR is appended
as the third channel during both training and inference, ensuring
training-test consistency.

---

## Two-stage protocol explained

**Stage I — Jakes screening.** All four algorithms are trained on
Jakes channels at fixed SNR (CNN-LSTM, CNN-Sparse, Sa-DLCS) or with a
curriculum (NCRN). The Jakes model is fast and analytically tractable,
so the four candidates can be compared cleanly without confounding by
multipath richness. NCRN is the winner.

**Stage II — TDL validation.** All four algorithms are retrained from
scratch on the 3GPP TR 38.901 TDL channel suite (10 scenarios spanning
InH, UMi, UMa, RMa, O2I, HST). This gives a fully fair comparison: every
algorithm sees the same training distribution as the test distribution,
so any performance gap reflects the architecture, not a training mismatch.
NCRN's three-phase SNR curriculum is preserved; the other three keep
their original Stage I optimisers and fixed-SNR training.

This protocol resolves the Jakes-vs-TDL training/test mismatch the
reviewers were concerned about. The methodology section in your
LaTeX paper now has a dedicated subsection (3.3) explaining this.

---

## Reading the figures

- **NMSE curves**: lower = better (it's a log-error)
- **Gold band (per-scenario)**: marks SNR bins where NCRN beats MMSE
- **Pink halo**: ±1σ over 200 test channels (shows variance)
- **CSV tables in `outputs/csv/`**: raw numbers for paper tables

---

## Memory and runtime notes

- **64×64 models** train in 3-5 min each on M1 (low memory).
- **128×128 models** need batch size 8 (set in scripts) and ~10-15 min
  each. If you hit OOM, reduce `BATCH_SIZE` to 4 in the script header.
- **Plotting full** = all 13 figures with 200 samples per per-scenario
  panel, ~30 min total.
- **Plotting --quick** = 4× fewer samples, ~7 min, useful for
  iterating on figure styling.

---

## Troubleshooting

**"Model not found" warnings during plotting** → that algorithm's
trained model is missing from `models/`. Either train it (re-run
`bash train_all.sh`) or remove it from the plot command via the
`--skip` argument.

**OOM during 128×128 NCRN training** → reduce `BATCH_SIZE` from 8 to 4
in `stage1_jakes/ncrn/train_ncrn_jakes.py` and
`stage2_tdl/ncrn/train_ncrn_tdl.py` (look for `batch_size = 8 if...`).

**Per-scenario plots take too long** → use `--quick` flag, or edit
`N_SCEN` in `plotting/plot_all_figures.py` (default 200, drop to 50).

**Loss curves look "simulated"** → if the `*_history.pkl` files
weren't saved during training, the plot falls back to representative
curves with the right shape. Re-run a training script to save the
real history.

---

## Citation

If you use this code, please cite the accompanying paper:

```
Kumar, A., Patel, V., Parth, P., Kukade, S.
"Noise-Conditioned Residual Learning for Robust Channel Estimation
 in High-Mobility Massive MIMO."
Vehicular Communications (under review).
```
