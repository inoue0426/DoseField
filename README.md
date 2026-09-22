# DoseField minimum dose-holdout experiment

This repository tests whether a dose-indexed latent vector field can interpolate a held-out 0.5 uM single-cell population when trained only at 0.05 and 5 uM. The field is the prescribed Euler model, `z_hat = z_0.05 + 0.5 v_theta(z_0.05, 0.25)`, with a two-layer 128-unit MLP, Adam, seed 0, and kinetic penalty `1e-3 ||v||^2`. Control cells are used only as a reference for differential-expression metrics.

The input is the public CC0 [Tahoe-100M dataset](https://huggingface.co/datasets/tahoebio/Tahoe-100M). The script verifies plate 13, downloads every third shard from 2274 through 2559, selects the highest-coverage cell line, and selects the top 20 drugs with at least 200 cells at each positive dose. It fits PCA-50 on log-normalized highly variable genes using control plus 0.05/5 uM cells only. The 0.5 uM cells are isolated until final evaluation.

## Reproduce

```bash
uv run --with huggingface_hub,pandas,pyarrow,numpy,scipy,scikit-learn,torch,pot python train_dose_field.py
```

The run writes `results.csv`, `results.md`, and `config.json`. Downloaded data are stored under `data/` and are intentionally gitignored. `config.json` records the selected drugs, cell counts, shard checksums, model settings, and training epochs.

For a specific already-verified cell line, pass `--cell-line CVCL_0459`; output names can be changed with `--results-name`, `--config-name`, and `--results-md-name`. The follow-up robustness check reused the existing plate-13 shards and screened only the second- and third-ranked lines by downloaded cell count.

Metrics are population mean cosine similarity, RBF MMD, and Jaccard overlap of top-100 absolute pseudobulk log-fold-change genes against the observed 0.5 uM population. OT interpolation uses POT's exact Earth Mover plan with uniform weights and McCann midpoint barycenters.
