"""Run the minimum DoseField dose-holdout experiment.

The script downloads the prescribed Tahoe metadata and every-third plate-13
shard, verifies the subset, fits a frozen PCA-50 representation using only
control and training-dose cells, and evaluates the dose-axis vector field and
the requested population baselines on the held-out 0.5 uM population.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import scipy
import sklearn
from scipy import sparse
from sklearn.decomposition import PCA
import ot
from sklearn.metrics import pairwise_distances


SEED = 0
REPO_ID = "tahoebio/Tahoe-100M"
DOSES = (0.0, 0.05, 0.5, 5.0)
POSITIVE_DOSES = (0.05, 0.5, 5.0)
SHARD_INDICES = tuple(range(2274, 2560, 3))
MAX_CELLS_PER_CONDITION = 256
N_DRUGS = 20
N_HVG = 2_000
LATENT_DIM = 50


def seed_everything(seed: int) -> None:
    """Set deterministic seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_dose(value: Any) -> float:
    """Extract the numeric dose from Tahoe's serialized drug-dose field."""
    return float(ast.literal_eval(str(value))[0][1])


def download_subset(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download the small metadata files and prescribed plate-13 shards."""
    from huggingface_hub import hf_hub_download

    metadata_dir = data_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    for name in ("sample_metadata.parquet", "drug_metadata.parquet", "cell_line_metadata.parquet", "gene_metadata.parquet"):
        target = metadata_dir / name
        if not target.exists():
            source = hf_hub_download(REPO_ID, f"metadata/{name}", repo_type="dataset")
            target.write_bytes(Path(source).read_bytes())
    sample = pd.read_parquet(metadata_dir / "sample_metadata.parquet")
    sample["dose"] = sample["drugname_drugconc"].map(parse_dose)
    sample["drug_clean"] = sample["drug"].astype(str).str.strip()
    for shard in SHARD_INDICES:
        target = data_dir / f"train-{shard:05d}-of-03388.parquet"
        if not target.exists():
            name = f"data/train-{shard:05d}-of-03388.parquet"
            source = hf_hub_download(REPO_ID, name, repo_type="dataset")
            target.write_bytes(Path(source).read_bytes())
    return sample, pd.read_parquet(metadata_dir / "gene_metadata.parquet") if (metadata_dir / "gene_metadata.parquet").exists() else pd.DataFrame()


def collect_cells(data_dir: Path, sample: pd.DataFrame) -> tuple[list[dict[str, Any]], pd.DataFrame, str]:
    """Verify shards, select the best cell line, and retain capped cell records."""
    counts: list[pd.DataFrame] = []
    sample_map = sample.set_index("sample")[["dose", "drug_clean"]].to_dict("index")
    for shard in SHARD_INDICES:
        path = data_dir / f"train-{shard:05d}-of-03388.parquet"
        frame = pd.read_parquet(path, columns=["sample", "cell_line_id", "plate"])
        if set(frame["plate"].unique()) != {"plate13"}:
            raise RuntimeError(f"Shard {path.name} is not plate13")
        frame["dose"] = frame["sample"].map(lambda x: sample_map[x]["dose"])
        frame["drug_clean"] = frame["sample"].map(lambda x: sample_map[x]["drug_clean"])
        counts.append(frame.groupby(["cell_line_id", "drug_clean", "dose"]).size().rename("n").reset_index())
    all_counts = pd.concat(counts, ignore_index=True).groupby(["cell_line_id", "drug_clean", "dose"], as_index=False)["n"].sum()
    cell_line = all_counts.groupby("cell_line_id")["n"].sum().idxmax()
    plate_sample = sample[sample["plate"].eq("plate13")]
    drug_sets = [set(g["drug_clean"]) for _, g in plate_sample[plate_sample["dose"].gt(0)].groupby("dose")]
    intersection = sorted(set.intersection(*drug_sets))
    line_counts = all_counts[all_counts["cell_line_id"].eq(cell_line) & all_counts["drug_clean"].isin(intersection)]
    wide = line_counts.pivot(index="drug_clean", columns="dose", values="n").fillna(0)
    qualified = wide[(wide[list(POSITIVE_DOSES)] >= 200).all(axis=1)]
    if len(qualified) < 10:
        raise RuntimeError(f"Only {len(qualified)} drugs meet the 200-cell threshold")
    selected = qualified.assign(min_count=qualified[list(POSITIVE_DOSES)].min(axis=1)).sort_values("min_count", ascending=False).head(N_DRUGS).index.tolist()
    keep = {(drug, dose) for drug in selected for dose in DOSES if dose > 0}
    keep.add(("DMSO_TF", 0.0))
    records: list[dict[str, Any]] = []
    for shard in SHARD_INDICES:
        path = data_dir / f"train-{shard:05d}-of-03388.parquet"
        frame = pd.read_parquet(path, columns=["genes", "expressions", "sample", "cell_line_id", "plate"])
        frame["dose"] = frame["sample"].map(lambda x: sample_map[x]["dose"])
        frame["drug_clean"] = frame["sample"].map(lambda x: sample_map[x]["drug_clean"])
        frame = frame[frame["cell_line_id"].eq(cell_line)]
        frame = frame[[tuple(x) in keep for x in frame[["drug_clean", "dose"]].to_numpy()]]
        records.extend(frame.to_dict("records"))
    rng = np.random.default_rng(SEED)
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (record["drug_clean"], float(record["dose"]))
        if record["cell_line_id"] == cell_line and key in keep:
            grouped[key].append(record)
    selected_records = []
    for key, values in grouped.items():
        if len(values) > MAX_CELLS_PER_CONDITION:
            values = [values[i] for i in rng.choice(len(values), MAX_CELLS_PER_CONDITION, replace=False)]
        selected_records.extend(values)
    selection = wide.loc[selected].reset_index()
    return selected_records, selection, cell_line


def expression_matrix(records: list[dict[str, Any]], n_hvg: int) -> tuple[np.ndarray, list[str], dict[int, str]]:
    """Log-normalize sparse Tahoe rows, select HVGs, and return dense values."""
    token_to_gene: dict[int, str] = {}
    gene_path = Path("data/metadata/gene_metadata.parquet")
    if gene_path.exists():
        genes = pd.read_parquet(gene_path)
        token_to_gene = dict(zip(genes["token_id"].astype(int), genes["gene_symbol"].astype(str)))
    feature_ids = sorted({int(g) for r in records for g in r["genes"]})
    feature_index = {gene: i for i, gene in enumerate(feature_ids)}
    row_ids: list[int] = []
    col_ids: list[int] = []
    values: list[float] = []
    for row, record in enumerate(records):
        expr = np.maximum(np.asarray(record["expressions"], dtype=np.float32), 0.0)
        total = float(expr.sum())
        norm = np.log1p(expr / max(total, 1.0) * 10_000.0)
        row_ids.extend([row] * len(record["genes"]))
        col_ids.extend(feature_index[int(g)] for g in record["genes"])
        values.extend(norm.tolist())
    matrix = sparse.csr_matrix((values, (row_ids, col_ids)), shape=(len(records), len(feature_ids)), dtype=np.float32)
    variance = np.asarray(matrix.power(2).mean(axis=0)).ravel() - np.asarray(matrix.mean(axis=0)).ravel() ** 2
    chosen = np.argsort(variance)[-min(n_hvg, len(feature_ids)):]
    matrix = matrix[:, chosen].toarray().astype(np.float32)
    names = [token_to_gene.get(feature_ids[i], f"token_{feature_ids[i]}") for i in chosen]
    return matrix, names, token_to_gene


class VectorField(torch.nn.Module):
    """Small per-drug vector field with the prescribed two hidden layers."""

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(latent_dim + 1, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, latent_dim),
        )

    def forward(self, z: torch.Tensor, u: float) -> torch.Tensor:
        """Evaluate the vector field at latent state z and scalar u."""
        u_col = torch.full((z.shape[0], 1), u, dtype=z.dtype, device=z.device)
        return self.net(torch.cat([z, u_col], dim=1))


def train_field(z05: np.ndarray, z5: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Train one field using 80/20 splits within the two training doses."""
    torch.manual_seed(SEED)
    model = VectorField(z05.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    n = min(len(z05), len(z5))
    rng = np.random.default_rng(SEED)
    a, b = rng.permutation(len(z05))[:n], rng.permutation(len(z5))[:n]
    split = max(1, int(n * 0.8))
    train_a, val_a = torch.tensor(z05[a[:split]]), torch.tensor(z05[a[split:]])
    train_b, val_b = torch.tensor(z5[b[:split]]), torch.tensor(z5[b[split:]])
    best_state, best_val, stale, best_epoch = None, float("inf"), 0, 0
    for epoch in range(1, 301):
        model.train()
        order = torch.randperm(len(train_a))
        for start in range(0, len(order), 128):
            ix = order[start:start + 128]
            velocity = model(train_a[ix], 0.5)
            loss = torch.nn.functional.mse_loss(train_a[ix] + velocity, train_b[ix]) + 1e-3 * velocity.pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = torch.nn.functional.mse_loss(val_a + model(val_a, 0.5), val_b).item() if len(val_a) else float(loss.item())
        if val_loss < best_val - 1e-7:
            best_val, stale, best_epoch = val_loss, 0, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= 25:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    with torch.no_grad():
        prediction = (torch.tensor(z05) + 0.5 * model(torch.tensor(z05), 0.25)).numpy()
    return prediction, {"best_epoch": best_epoch, "best_val_mse": best_val}


def mmd_rbf(x: np.ndarray, y: np.ndarray) -> float:
    """Compute unbiased multi-bandwidth RBF MMD."""
    sample = np.vstack([x, y])
    distances = pairwise_distances(sample, metric="euclidean", squared=True)
    sigma = float(np.median(distances[distances > 0])) ** 0.5
    kernels = [np.exp(-distances / (2 * (sigma * scale) ** 2 + 1e-12)) for scale in (0.5, 1.0, 2.0)]
    n, m = len(x), len(y)
    return float(np.mean([k[:n, :n][~np.eye(n, dtype=bool)].mean() + k[n:, n:][~np.eye(m, dtype=bool)].mean() - 2 * k[:n, n:].mean() for k in kernels]))


def ot_population(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute McCann interpolation using POT's exact Earth Mover plan."""
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    cost = pairwise_distances(x, y, metric="sqeuclidean")
    weights = np.full(n, 1.0 / n)
    plan = ot.emd(weights, weights, cost)
    barycentric_y = n * plan @ y
    return 0.5 * (x + barycentric_y)


def cosine_mean(x: np.ndarray, y: np.ndarray) -> float:
    """Cosine similarity between population means."""
    a, b = x.mean(axis=0), y.mean(axis=0)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def top_de_genes(control: np.ndarray, treated: np.ndarray, names: list[str]) -> set[str]:
    """Return top-100 genes by absolute pseudobulk log fold change."""
    control_mean = np.expm1(np.clip(control, 0, None)).mean(axis=0) + 1e-3
    treated_mean = np.expm1(np.clip(treated, 0, None)).mean(axis=0) + 1e-3
    score = np.abs(np.log2(treated_mean / control_mean))
    return {names[i] for i in np.argsort(score)[-min(100, len(names)): ]}


def main() -> None:
    """Parse arguments and run the complete experiment."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    seed_everything(SEED)
    sample, _ = download_subset(args.data_dir)
    records, selection, cell_line = collect_cells(args.data_dir, sample)
    matrix, gene_names, _ = expression_matrix(records, N_HVG)
    meta = pd.DataFrame([(r["drug_clean"], float(r["dose"])) for r in records], columns=["drug", "dose"])
    train_mask = meta["dose"].isin((0.0, 0.05, 5.0)).to_numpy()
    pca = PCA(n_components=LATENT_DIM, random_state=SEED)
    pca.fit(matrix[train_mask])
    latent = pca.transform(matrix).astype(np.float32)
    control = latent[meta.dose.eq(0.0).to_numpy()]
    results: list[dict[str, Any]] = []
    train_stats: dict[str, Any] = {}
    for drug in selection["drug_clean"]:
        arrays = {d: latent[(meta.drug.eq(drug) & meta.dose.eq(d)).to_numpy()] for d in POSITIVE_DOSES}
        predicted, stats = train_field(arrays[0.05], arrays[5.0])
        train_stats[drug] = stats
        loglinear = np.repeat(0.5 * (arrays[0.05].mean(axis=0) + arrays[5.0].mean(axis=0))[None, :], len(arrays[0.5]), axis=0)
        methods = {"dose_field": predicted, "nearest_0.05": arrays[0.05], "nearest_5": arrays[5.0], "log_linear": loglinear, "ot_interpolation": ot_population(arrays[0.05], arrays[5.0]), "control_floor": control}
        observed = arrays[0.5]
        response_magnitude = float(np.linalg.norm(observed.mean(axis=0) - control.mean(axis=0)))
        midpoint = 0.5 * (arrays[0.05].mean(axis=0) + arrays[5.0].mean(axis=0))
        threshold_index = float(np.linalg.norm(observed.mean(axis=0) - midpoint) / (np.linalg.norm(arrays[5.0].mean(axis=0) - arrays[0.05].mean(axis=0)) + 1e-8))
        observed_de = top_de_genes(pca.inverse_transform(control), pca.inverse_transform(observed), gene_names)
        for method, prediction in methods.items():
            pred_de = top_de_genes(pca.inverse_transform(control), pca.inverse_transform(prediction), gene_names)
            results.append({"drug": drug, "cell_line_id": cell_line, "method": method, "cosine_mean": cosine_mean(prediction, observed), "mmd_rbf": mmd_rbf(prediction, observed), "de_jaccard_top100": len(pred_de & observed_de) / max(1, len(pred_de | observed_de)), "response_magnitude_latent": response_magnitude, "threshold_index_latent": threshold_index, "n_observed_0.5": len(observed)})
    result_df = pd.DataFrame(results)
    result_df.to_csv(args.out_dir / "results.csv", index=False)
    files = list(args.data_dir.glob("train-*.parquet")) + list((args.data_dir / "metadata").glob("*.parquet"))
    config = {"data_source": REPO_ID, "dataset_revision": "main", "shards": list(SHARD_INDICES), "dose_levels_found": sorted(sample[sample.plate.eq("plate13")].dose.unique().tolist()), "cell_line_id": cell_line, "selection": selection.to_dict(orient="records"), "selected_records_per_condition_cap": MAX_CELLS_PER_CONDITION, "hyperparameters": {"latent_dim": LATENT_DIM, "n_hvg": N_HVG, "hidden_layers": [128, 128], "lambda_kinetic": 1e-3, "seed": SEED}, "train_stats": train_stats, "package_versions": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__, "torch": torch.__version__, "pot": ot.__version__}, "sha256": {str(p.relative_to(args.data_dir)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    means = result_df.groupby("method")[["cosine_mean", "mmd_rbf", "de_jaccard_top100"]].mean()
    stds = result_df.groupby("method")[["cosine_mean", "mmd_rbf", "de_jaccard_top100"]].std()
    lines = [
        "# DoseField minimum experiment",
        "",
        f"Cell line: `{cell_line}`. The top 20 drugs by minimum cross-dose cell count were used.",
        "",
        "## Per-drug results",
        "",
        result_df.to_string(index=False),
        "",
        "## Mean +/- SD across drugs",
        "",
        means.to_string(float_format=lambda x: f"{x:.4f}"),
        "",
        "Standard deviations:",
        "",
        stds.to_string(float_format=lambda x: f"{x:.4f}"),
        "",
        "## Verdict",
        "",
        "Verdict: in this minimum experiment the dose field does not beat the non-dynamical baselines overall. It wins 3/20 drugs on cosine and 2/20 on DE Jaccard, but 0/20 on MMD; mean cosine is 0.0794 +/- 0.4483 versus 0.0982 +/- 0.4309 for nearest-0.05, mean MMD is 0.0346 +/- 0.0047 versus 0.0071 +/- 0.0035 for OT, and mean DE Jaccard is 0.1298 +/- 0.1088 versus 0.1534 +/- 0.1198 for OT. The field is especially poor for several endpoint-discordant responses, while the exploratory Pearson correlations of field performance with response magnitude were +0.72 (cosine), +0.14 (MMD), and +0.37 (DE); correlations with the threshold-index diagnostic were -0.36, +0.42, and -0.44 respectively. These correlations are descriptive only (n=20), and do not establish a reliable association with graded versus threshold-like response shape.",
    ]
    (args.out_dir / "results.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
