"""
================================================================================
Task-State K Comparison: pre_model vs post_model
================================================================================

Runs the DESCRIPTIVE K analysis (eigenvalue table, spectrum, mode maps,
region-to-region block coupling) separately on each of the two task-state
models, saving each model's outputs to its own folder, then produces the
ONE genuinely comparative output: the block-coupling difference (post - pre).

WHY ONLY BLOCK COUPLING IS COMPARED (not eigenvalues mode-for-mode or mode
maps): pre_model and post_model are two SEPARATELY-TRAINED models, so their
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

IMPORTANT CAVEAT (not a code issue -- a claim-strength issue): with ONE model
per condition, any B difference confounds "K genuinely differs pre vs post"
with "two separate optimization runs landed differently." This is DESCRIPTIVE.
A real test of whether K differs would need multiple training runs per
condition (different seeds) to establish run-to-run variability.

No edits to compare_pre_post.py are needed -- its individual compute/plot
functions are imported and reused; only plot_mode_maps writes a CSV to that
module's global RESULTS_DIR, so that global is temporarily pointed at the
right folder around each call (scoped, restored after).

Usage:
    python analysis/compare_K_task_states.py
    python analysis/compare_K_task_states.py --top-k 20
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config import M, N_ROIS, H
from preprocessing.load_preprocessed_data import TARGET_ROIS
from analysis.analysis_helper_functions import load_model, compute_K

# Import existing plotting function from loso_k_analysis
from analysis.loso_k_analysis import plot_mode_index_heatmap

# Reuse the descriptive functions from compare_pre_post.py unchanged.
import analysis.compare_pre_post as cpp
from analysis.compare_pre_post import (
    eigenvalue_table, plot_spectrum, plot_mode_maps,
    compute_block_norms, plot_block_coupling,
)

# ================================================================================
# CONFIG -- output locations
# ================================================================================
TASK_STATES_TRAIN = ROOT_DIR / "results" / "training" / "task_states"
OUT_BASE = ROOT_DIR / "results" / "figures" / "model_plots" / "task_states"

MODELS = {
    "pre_model":  TASK_STATES_TRAIN / "pre_model"  / "best_model_cls.pt",
    "post_model": TASK_STATES_TRAIN / "post_model" / "best_model_cls.pt",
}
COMPARISON_DIR = OUT_BASE / "K_comparison"


# ================================================================================
# PER-MODEL DESCRIPTIVES
# ================================================================================
def run_descriptives(model, out_dir: Path, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Runs the descriptive K outputs for one model into out_dir.
    Returns:
        B: block-coupling matrix (N_ROIS x N_ROIS)
        sorted_mode_indices: array of mode indices sorted by persistence (|Lambda| desc)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n--- K descriptives -> {out_dir} ---")

    K, Lambda, W_bar_x = compute_K(model)

    # Eigenvalue table
    eig_df = eigenvalue_table(Lambda)
    eig_path = out_dir / "koopman_eigenvalues.csv"
    eig_df.to_csv(eig_path, index=False)
    print(f"Saved {eig_path}")
    print(f"  Persistent modes (|\u039b|>0.9): {(np.abs(Lambda) > 0.9).sum()} / {len(Lambda)}")
    print(f"  Spectral radius (max|\u039b|): {np.abs(Lambda).max():.4f}")

    # Compute mode indices sorted by persistence (|Lambda| descending)
    sorted_mode_indices = np.argsort(-np.abs(Lambda))

    # Spectrum
    plot_spectrum(Lambda, out_dir / "koopman_spectrum.png")

    # Mode maps
    saved_results_dir = cpp.RESULTS_DIR
    cpp.RESULTS_DIR = out_dir
    try:
        plot_mode_maps(W_bar_x, Lambda, TARGET_ROIS,
                       out_dir / "koopman_mode_maps.png", top_k=top_k)
    finally:
        cpp.RESULTS_DIR = saved_results_dir

    # Block coupling B
    B = compute_block_norms(K, N_ROIS, H)
    B_df = pd.DataFrame(B, index=list(TARGET_ROIS), columns=list(TARGET_ROIS))
    B_df.index.name = "target_ROI"     # rows = target (t+1)
    B_df.columns.name = "source_ROI"   # cols = source (t)
    B_path = out_dir / "K_region_coupling.csv"
    B_df.to_csv(B_path)
    print(f"Saved {B_path}")
    plot_block_coupling(B, TARGET_ROIS, out_dir / "K_region_coupling.png")

    return B, sorted_mode_indices


# ================================================================================
# BLOCK-COUPLING DIFFERENCE (post - pre) 
# ================================================================================
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


# ================================================================================
# MAIN
# ================================================================================
def main(top_k: int = M):
    print("=" * 64)
    print("Task-State K Comparison: pre_model vs post_model")
    print("=" * 64)

    # Guard: both checkpoints must exist before doing anything.
    for name, path in MODELS.items():
        if not path.exists():
            raise FileNotFoundError(f"{name} checkpoint not found: {path}")

    B_by_model = {}

    rank_indices_by_model = []
    model_names = list(MODELS.keys())

    for name, ckpt_path in MODELS.items():
        print(f"\n=== {name} ===")
        model = load_model(ckpt_path)
        out_dir = OUT_BASE / name / "K_descriptives"
        B, sorted_indices = run_descriptives(model, out_dir, top_k=top_k)

        B_by_model[name] = B
        rank_indices_by_model.append(sorted_indices)

    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    idx_mat = np.vstack(rank_indices_by_model)  # Shape: (2, M)
    
    # Direct reuse of plot_mode_index_heatmap from analysis.loso_k_analysis
    plot_mode_index_heatmap(
        idx_mat=idx_mat,
        fold_ids=model_names,  # Passes ['pre_model', 'post_model'] for row labels
        M=M,
        out_path=COMPARISON_DIR / "mode_persistence_rank_heatmap.png"
    )


    # --- Comparative output: block-coupling difference (post - pre) ---
    B_pre = B_by_model["pre_model"]
    B_post = B_by_model["post_model"]
    B_diff = B_post - B_pre

    diff_df = pd.DataFrame(B_diff, index=list(TARGET_ROIS), columns=list(TARGET_ROIS))
    diff_df.index.name = "target_ROI"
    diff_df.columns.name = "source_ROI"
    diff_path = COMPARISON_DIR / "K_region_coupling_diff.csv"
    diff_df.to_csv(diff_path)
    print(f"\nSaved {diff_path}")

    plot_block_coupling_diff(B_diff, TARGET_ROIS,
                             COMPARISON_DIR / "K_region_coupling_diff.png")

    # Quick scalar summary of the difference.
    print(f"\nBlock-coupling difference summary (post - pre):")
    print(f"  max increase (post>pre): {B_diff.max():+.4f}")
    print(f"  max decrease (pre>post): {B_diff.min():+.4f}")
    print(f"  mean |diff|:             {np.abs(B_diff).mean():.4f}")
    print(f"  Frobenius norm of diff:  {np.linalg.norm(B_diff):.4f}")
    print("\n  (DESCRIPTIVE -- one model per condition; a difference here "
          "cannot be separated from run-to-run optimization variability.)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=M,
                        help="Number of most-persistent modes to plot in mode maps.")
    args = parser.parse_args()
    main(top_k=args.top_k)