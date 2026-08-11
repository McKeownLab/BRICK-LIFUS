# analysis/task_state_variance_decomposition.py
"""
================================================================================
Task-State Analysis: pre_model vs post_model
================================================================================

Two analyses of two task-state models (pre_model, post_model; see
train_task_states.py), sharing one output directory:

    PART 1: instantaneous-C vs accumulated-K variance decomposition, per
    subject, aggregated with an explicit in-sample/held-out split.

    PART 2: descriptive K comparison (eigenvalue spectrum, mode maps,
    region-to-region block coupling) run separately per model, plus the one
    genuinely cross-model comparison -- block-coupling difference (post -
    pre). See PART 2 docstring below for why eigenvalues/mode maps are
    NOT compared mode-for-mode across the two models.

Usage:
    python analysis/task_state_variance_decomposition.py
    python analysis/task_state_variance_decomposition.py --top-k 20
"""

import sys
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import config
from config import M, N_ROIS, H
from preprocessing.load_preprocessed_data import load_all, TARGET_ROIS
from analysis.analysis_helper_functions import load_model, compute_K
from analysis.C_stability_perturbation_analysis import (
    encode_and_control, manual_rollout, variance_decomposition, to_complex_tensor,
)
from training.train_task_states import select_val_subjects

# Reuse descriptive K functions from compare_pre_post.py unchanged.
import analysis.compare_pre_post as cpp
from analysis.compare_pre_post import (
    eigenvalue_table, plot_spectrum, plot_mode_maps,
    compute_block_norms, plot_block_coupling,
)
from analysis.K_loso_analysis import plot_mode_index_heatmap

TASK_STATES_TRAIN = ROOT_DIR / "results" / "training" / "task_states"
OUT_DIR = ROOT_DIR / "results" / "figures" / "model_plots" / "task_states"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "pre_model":  {"checkpoint": TASK_STATES_TRAIN / "pre_model"  / "best_model_cls.pt", "session_key": "mpre"},
    "post_model": {"checkpoint": TASK_STATES_TRAIN / "post_model" / "best_model_cls.pt", "session_key": "mpost"},
}
K_COMPARISON_DIR = OUT_DIR / "K_comparison"


# ================================================================================
# PART 1: TASK-STATE VARIANCE DECOMPOSITION (pre_model vs post_model)
# ================================================================================
"""
CRITICAL CAVEAT -- NOT a LOSO-style held-out evaluation:
    train_task_states.py trains ONE pre_model and ONE post_model, each on ALL
    subjects except 2 held out for validation (same 2 subjects for both
    models; no test set; no per-subject fold). For 17 of 19 subjects, the
    numbers below are IN-SAMPLE -- a fundamentally weaker claim than a
    genuinely held-out figure would be.

    This section reports THREE tiers:
        1. ALL subjects (17 train + 2 val) -- most data, weakest claim.
        2. VAL-ONLY (2 subjects) -- genuinely held-out, but tiny N; a sanity
           check, not a robust estimate on its own.
        3. TRAIN-ONLY (17 subjects) -- in-sample by construction; shown only
           for comparison against (2), never as standalone evidence.
    val/train subject IDs are recovered via train_task_states.py's own
    select_val_subjects(), so the split matches training exactly.
"""


def znorm_np(x: np.ndarray) -> torch.Tensor:
    """Per-ROI z-score, matching TaskStateDataset's normalization exactly."""
    x = torch.tensor(x, dtype=torch.float32)
    return (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-8)


def compute_subject_decomposition(model, Lambda, P_inv, W_bar_x, x):
    """One subject-target entry: encode + control (unperturbed), manual
    rollout, variance decomposition. Returns list of per-ROI dicts."""
    g_0, C_base, u = encode_and_control(model, x)
    x_hat, x_K, x_C = manual_rollout(
        Lambda, P_inv, g_0, u, W_bar_x, C_seq=C_base, return_decomposition=True
    )
    return variance_decomposition(x_K.numpy(), x_C.numpy())


def aggregate_rows(all_rows: list) -> dict:
    """Variance-weighted pooling across every (subject, ROI) row."""
    per_roi = defaultdict(list)
    for r in all_rows:
        per_roi[r["roi"]].append(r)

    roi_summary = []
    for roi, rows in per_roi.items():
        var_total = sum(r["var_total"] for r in rows)
        pct_c = sum(r["pct_instantaneous_C"] * r["var_total"] for r in rows) / var_total
        pct_k = sum(r["pct_accumulated_K"] * r["var_total"] for r in rows) / var_total
        roi_summary.append({"roi": roi, "n_entries": len(rows),
                             "pct_instantaneous_C": pct_c, "pct_accumulated_K": pct_k,
                             "var_total": var_total})

    overall_var_total = sum(r["var_total"] for r in all_rows)
    overall_pct_c = sum(r["pct_instantaneous_C"] * r["var_total"] for r in all_rows) / overall_var_total
    overall_pct_k = sum(r["pct_accumulated_K"] * r["var_total"] for r in all_rows) / overall_var_total

    return {
        "per_roi": sorted(roi_summary, key=lambda r: r["roi"]),
        "overall_pct_instantaneous_C": overall_pct_c,
        "overall_pct_accumulated_K": overall_pct_k,
        "n_rows": len(all_rows),
    }


def print_tier(label: str, agg: dict):
    print(f"\n--- {label} ---")
    print(f"  {'ROI':<30} {'% instant. (C)':>15} {'% accum. (K)':>14} {'n_entries':>10}")
    for r in agg["per_roi"]:
        print(f"  {r['roi']:<30} {r['pct_instantaneous_C']:>14.1f}% "
              f"{r['pct_accumulated_K']:>13.1f}% {r['n_entries']:>10d}")
    print(f"  {'VARIANCE-WEIGHTED MEAN':<30} {agg['overall_pct_instantaneous_C']:>14.1f}% "
          f"{agg['overall_pct_accumulated_K']:>13.1f}%")


def run_variance_decomposition_for_model(name: str, checkpoint_path: Path,
                                          session_key: str, val_subjects: set):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"{name} checkpoint not found: {checkpoint_path}")

    print("=" * 64)
    print(f"{name}  (session={session_key}, checkpoint={checkpoint_path})")
    print("=" * 64)

    model = load_model(checkpoint_path)
    K, Lambda, W_bar_x = compute_K(model)
    Lambda = to_complex_tensor(Lambda)
    W_bar_x = to_complex_tensor(W_bar_x)
    P_inv = model.P_inv

    subjects = load_all()

    all_rows, val_rows, train_rows = [], [], []
    for s in subjects:
        x = znorm_np(s[session_key])
        rows = compute_subject_decomposition(model, Lambda, P_inv, W_bar_x, x)
        for r in rows:
            r["subject_id"] = s["subject_id"]
            r["target"] = s["target"]
        all_rows.extend(rows)
        if s["subject_id"] in val_subjects:
            val_rows.extend(rows)
        else:
            train_rows.extend(rows)

    n_subjects = len(set(s["subject_id"] for s in subjects))
    n_val = len(val_subjects)
    n_train = n_subjects - n_val
    print(f"\n{n_subjects} subjects total ({len(subjects)} subject-target entries): "
          f"{n_train} in-sample (train), {n_val} held-out (val: {sorted(val_subjects)})")

    agg_all = aggregate_rows(all_rows)
    agg_val = aggregate_rows(val_rows) if val_rows else None
    agg_train = aggregate_rows(train_rows) if train_rows else None

    print_tier(f"{name} -- ALL subjects (in-sample + held-out mixed; weakest claim)", agg_all)
    if agg_val:
        print_tier(f"{name} -- VAL-ONLY (genuinely held-out, n={n_val} subjects, small N)", agg_val)
    if agg_train:
        print_tier(f"{name} -- TRAIN-ONLY (in-sample by construction; NOT generalization evidence)", agg_train)

    pd.DataFrame(all_rows).to_csv(OUT_DIR / f"{name}_variance_decomposition_raw.csv", index=False)
    pd.DataFrame(agg_all["per_roi"]).to_csv(OUT_DIR / f"{name}_variance_decomposition_all.csv", index=False)
    if agg_val:
        pd.DataFrame(agg_val["per_roi"]).to_csv(OUT_DIR / f"{name}_variance_decomposition_val_only.csv", index=False)
    if agg_train:
        pd.DataFrame(agg_train["per_roi"]).to_csv(OUT_DIR / f"{name}_variance_decomposition_train_only.csv", index=False)

    return {"all": agg_all, "val": agg_val, "train": agg_train}


def run_task_state_variance_decomposition():
    val_subjects = set(select_val_subjects(condition="pre", seed=config.SEED))
    print(f"Held-out val subjects (shared across both models): {sorted(val_subjects)}\n")

    results = {}
    for name, info in MODELS.items():
        results[name] = run_variance_decomposition_for_model(
            name, info["checkpoint"], info["session_key"], val_subjects
        )

    print("\n" + "=" * 64)
    print("SUMMARY -- variance-weighted mean, % instantaneous C vs % accumulated K")
    print("=" * 64)
    print(f"{'':<12} {'ALL':>22} {'VAL-ONLY':>22} {'TRAIN-ONLY':>22}")

    def fmt(agg):
        if agg is None:
            return "n/a"
        return f"{agg['overall_pct_instantaneous_C']:.1f}% C / {agg['overall_pct_accumulated_K']:.1f}% K"

    summary_rows = []
    for name in MODELS:
        r = results[name]
        print(f"{name:<12} {fmt(r['all']):>22} {fmt(r['val']):>22} {fmt(r['train']):>22}")
        summary_rows.append({
            "model": name,
            "all_pct_C": r["all"]["overall_pct_instantaneous_C"],
            "all_pct_K": r["all"]["overall_pct_accumulated_K"],
            "val_pct_C": r["val"]["overall_pct_instantaneous_C"] if r["val"] else np.nan,
            "val_pct_K": r["val"]["overall_pct_accumulated_K"] if r["val"] else np.nan,
            "train_pct_C": r["train"]["overall_pct_instantaneous_C"] if r["train"] else np.nan,
            "train_pct_K": r["train"]["overall_pct_accumulated_K"] if r["train"] else np.nan,
        })
    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "summary.csv", index=False)
    print(f"\nSaved {OUT_DIR / 'summary.csv'}")


# ================================================================================
# PART 2: DESCRIPTIVE K COMPARISON (pre_model vs post_model)
# ================================================================================
"""
Runs the DESCRIPTIVE K analysis (eigenvalue table, spectrum, mode maps,
region-to-region block coupling) separately on each of the two task-state
models, saving each model's outputs to its own folder, then produces the
ONE comparative output: the block-coupling difference (post - pre).

pre_model and post_model are two SEPARATELY-TRAINED models, so their
eigenbases are not aligned -- "mode M43" means different things in each, and
eigenvector signs/phases plus the within-ROI H-channel basis are arbitrary
per model. So:
    - eigenvalue TABLE / spectrum : basis-free scalars (spectral radius,
      #persistent modes, freq distribution) -- comparable by eye across the
      two per-model outputs, but NOT paired mode-for-mode.
    - mode maps                   : per-model only, NOT cross-comparable.
    - block coupling B            : ROI-block Frobenius norms are ~rotation-
      invariant (ROI-block identity is fixed by ROI-major construction), so
      B IS comparable across independently-trained models. The post - pre
      difference is therefore the defensible "did region-to-region dynamics
      shift" view.
"""


def run_K_descriptives_for_model(model, out_dir: Path, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Runs the descriptive K outputs for one model into out_dir.
    Returns:
        B: block-coupling matrix (N_ROIS x N_ROIS)
        sorted_mode_indices: array of mode indices sorted by persistence (|Lambda| desc)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n--- K descriptives -> {out_dir} ---")

    K, Lambda, W_bar_x = compute_K(model)

    eig_df = eigenvalue_table(Lambda)
    eig_path = out_dir / "koopman_eigenvalues.csv"
    eig_df.to_csv(eig_path, index=False)
    print(f"Saved {eig_path}")
    print(f"  Persistent modes (|\u039b|>0.9): {(np.abs(Lambda) > 0.9).sum()} / {len(Lambda)}")
    print(f"  Spectral radius (max|\u039b|): {np.abs(Lambda).max():.4f}")

    sorted_mode_indices = np.argsort(-np.abs(Lambda))

    plot_spectrum(Lambda, out_dir / "koopman_spectrum.png")

    saved_results_dir = cpp.RESULTS_DIR
    cpp.RESULTS_DIR = out_dir
    try:
        plot_mode_maps(W_bar_x, Lambda, TARGET_ROIS,
                       out_dir / "koopman_mode_maps.png", top_k=top_k)
    finally:
        cpp.RESULTS_DIR = saved_results_dir

    B = compute_block_norms(K, N_ROIS, H)
    B_df = pd.DataFrame(B, index=list(TARGET_ROIS), columns=list(TARGET_ROIS))
    B_df.index.name = "target_ROI"
    B_df.columns.name = "source_ROI"
    B_path = out_dir / "K_region_coupling.csv"
    B_df.to_csv(B_path)
    print(f"Saved {B_path}")
    plot_block_coupling(B, TARGET_ROIS, out_dir / "K_region_coupling.png")

    return B, sorted_mode_indices


def plot_block_coupling_diff(B_diff: np.ndarray, roi_names, out_path: Path):
    """Diverging heatmap of B_post - B_pre, symmetric about 0."""
    vmax = np.abs(B_diff).max()
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(B_diff, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="\u0394 block coupling norm (post \u2212 pre)")
    ax.set_xticks(range(len(roi_names)))
    ax.set_yticks(range(len(roi_names)))
    ax.set_xticklabels(roi_names, rotation=90, fontsize=7)
    ax.set_yticklabels(roi_names, fontsize=7)
    ax.set_xlabel("source ROI (t)")
    ax.set_ylabel("target ROI (t+1)")
    ax.set_title("\u0394 region-to-region latent coupling  (post \u2212 pre)\n"
                 "red = stronger post, blue = stronger pre  |  DESCRIPTIVE "
                 "(two separately-trained models; one run each)",
                 fontsize=10)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


def run_K_comparison(top_k: int):
    print("=" * 64)
    print("Task-State K Comparison: pre_model vs post_model")
    print("=" * 64)

    for name, info in MODELS.items():
        if not info["checkpoint"].exists():
            raise FileNotFoundError(f"{name} checkpoint not found: {info['checkpoint']}")

    B_by_model = {}
    rank_indices_by_model = []
    model_names = list(MODELS.keys())

    for name, info in MODELS.items():
        print(f"\n=== {name} ===")
        model = load_model(info["checkpoint"])
        out_dir = OUT_DIR / name / "K_descriptives"
        B, sorted_indices = run_K_descriptives_for_model(model, out_dir, top_k=top_k)

        B_by_model[name] = B
        rank_indices_by_model.append(sorted_indices)

    K_COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    idx_mat = np.vstack(rank_indices_by_model)

    plot_mode_index_heatmap(
        idx_mat=idx_mat,
        fold_ids=model_names,
        M=M,
        out_path=K_COMPARISON_DIR / "mode_persistence_rank_heatmap.png"
    )

    B_pre = B_by_model["pre_model"]
    B_post = B_by_model["post_model"]
    B_diff = B_post - B_pre

    diff_df = pd.DataFrame(B_diff, index=list(TARGET_ROIS), columns=list(TARGET_ROIS))
    diff_df.index.name = "target_ROI"
    diff_df.columns.name = "source_ROI"
    diff_path = K_COMPARISON_DIR / "K_region_coupling_diff.csv"
    diff_df.to_csv(diff_path)
    print(f"\nSaved {diff_path}")

    plot_block_coupling_diff(B_diff, TARGET_ROIS,
                             K_COMPARISON_DIR / "K_region_coupling_diff.png")

    print(f"\nBlock-coupling difference summary (post - pre):")
    print(f"  max increase (post>pre): {B_diff.max():+.4f}")
    print(f"  max decrease (pre>post): {B_diff.min():+.4f}")
    print(f"  mean |diff|:             {np.abs(B_diff).mean():.4f}")
    print(f"  Frobenius norm of diff:  {np.linalg.norm(B_diff):.4f}")
    print("\n  (DESCRIPTIVE -- one model per condition; a difference here "
          "cannot be separated from run-to-run optimization variability.)")


# ================================================================================
# MAIN
# ================================================================================

def main(top_k: int = M):
    print("#" * 64)
    print("# PART 1: Task-state variance decomposition (pre_model / post_model)")
    print("#" * 64)
    run_task_state_variance_decomposition()

    print("\n\n" + "#" * 64)
    print("# PART 2: Descriptive K comparison (pre_model / post_model)")
    print("#" * 64)
    run_K_comparison(top_k=top_k)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=M,
                        help="Number of most-persistent modes to plot in mode maps (Part 2).")
    args = parser.parse_args()
    main(top_k=args.top_k)