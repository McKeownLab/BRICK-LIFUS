"""
================================================================================
Variance Decomposition — Task-State Models (pre_model / post_model)
================================================================================

Prints a summary table of %C (instantaneous injection) vs %K (accumulated
propagation) for the two task-state models, pooled across all 19 subjects,
alongside the original single-subject LOSO-fold result for comparison.

POOLING METHOD:
    For each subject, this script runs the model's encoder + control
    module on that subject's session, decomposes x_hat into x_K and x_C via
    manual_rollout(..., return_decomposition=True), and computes that
    subject's own per-ROI variance decomposition via variance_decomposition().

    All subjects' (ROI, var_K, var_C, cov_KC, var_total) rows are then
    pooled into ONE variance-weighted mean across subjects x ROIs -- i.e.
    each (subject, ROI) row contributes to the final %C / %K in proportion
    to its own var_total, using the same weighting convention as
    print_variance_decomposition()'s existing across-ROI mean, just
    extended to also pool across subjects. This is NOT the same as
    averaging each subject's own summary percentage unweighted -- a
    subject/ROI with more total variance has proportionally more say in
    the final number, consistent with how the original single-subject
    table's own ROI-level mean is computed.

Usage:
    python analysis/variance_decomposition_task_states.py
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import training.train as _train_module
import config as _config
_train_module.SEED = _config.SEED

import numpy as np
import torch

from analysis.analysis_helper_functions import load_model, compute_K
from analysis.C_stability_perturbation_analysis import (
    encode_and_control, manual_rollout, variance_decomposition, to_complex_tensor,
)
from training.train_task_states import TaskStateDataset
from preprocessing.load_preprocessed_data import TARGET_ROIS

# Change directory
TASK_STATES_DIR = ROOT_DIR / "results" / "training" / "task_states"

# Original single-subject result, already confirmed -- hardcoded here only
# for side-by-side printing, NOT recomputed by this script.
ORIGINAL_PCT_C = 94.2
ORIGINAL_PCT_K = 5.8
ORIGINAL_LABEL = "Original (1 subject, held-out)"


def run_pooled_decomposition(checkpoint_path: Path, condition: str, label: str) -> dict:
    """
    Loads one task-state model, runs the unperturbed forward pass for every
    subject in that condition's TaskStateDataset, and returns the
    variance-weighted mean %C / %K pooled across all (subject, ROI) rows,
    plus the raw per-subject-per-ROI rows for per-ROI reporting.
    """
    print(f"\nLoading {label} from {checkpoint_path} ...")
    model = load_model(checkpoint_path)
    K, Lambda, W_bar_x = compute_K(model)
    Lambda  = to_complex_tensor(Lambda)
    W_bar_x = to_complex_tensor(W_bar_x)
    P_inv = model.P_inv

    ds = TaskStateDataset(condition)
    print(f"  {len(ds.items)} subjects in {condition} condition")

    all_rows = []
    for item in ds.items:
        x = item["x"]
        g_0, C_base, u = encode_and_control(model, x)
        x_hat, x_K, x_C = manual_rollout(
            Lambda, P_inv, g_0, u, W_bar_x, C_seq=C_base, return_decomposition=True
        )
        subject_rows = variance_decomposition(x_K.numpy(), x_C.numpy())
        for r in subject_rows:
            r["subject_id"] = item["subject_id"]
        all_rows.extend(subject_rows)

    pooled_var_total = sum(r["var_total"] for r in all_rows)
    pooled_pct_c = sum(r["pct_instantaneous_C"] * r["var_total"] for r in all_rows) / pooled_var_total
    pooled_pct_k = sum(r["pct_accumulated_K"]   * r["var_total"] for r in all_rows) / pooled_var_total

    return {
        "label": label,
        "pct_C": pooled_pct_c,
        "pct_K": pooled_pct_k,
        "n_subjects": len(ds.items),
        "n_rows": len(all_rows),
        "all_rows": all_rows,
    }

def print_per_roi_table(result: dict):
    """
    Per-ROI mean %C/%K, averaged across subjects (unweighted mean of each
    subject's own per-ROI percentage for that ROI -- NOT variance-weighted,
    since the point here is "what does a typical subject look like at this
    ROI", not "how much of total variance". Use the pooled variance-weighted
    number from print_summary_table for the headline pooled result instead.
    """
    by_roi = {}
    for r in result["all_rows"]:
        by_roi.setdefault(r["roi"], []).append(r)

    print(f"\n{result['label']} -- per-ROI mean %C / %K "
          f"(unweighted mean across {result['n_subjects']} subjects):")
    print(f"  {'ROI':<30} {'% instant. (C)':>15} {'% accum. (K)':>14}")
    for roi_name in TARGET_ROIS:
        rows = by_roi.get(roi_name, [])
        if not rows:
            continue
        mean_c = np.mean([r["pct_instantaneous_C"] for r in rows])
        mean_k = np.mean([r["pct_accumulated_K"] for r in rows])
        print(f"  {roi_name:<30} {mean_c:>14.1f}% {mean_k:>13.1f}%")


def print_summary_table(results: list):
    print("\n" + "=" * 60)
    print(f"{'':<32} {'%C':>8} {'%K':>8}")
    print("-" * 60)
    print(f"{ORIGINAL_LABEL:<32} {ORIGINAL_PCT_C:>7.1f}% {ORIGINAL_PCT_K:>7.1f}%")
    for r in results:
        row_label = f"{r['label']} (all {r['n_subjects']})"
        print(f"{row_label:<32} {r['pct_C']:>7.1f}% {r['pct_K']:>7.1f}%")
    print("=" * 60)


def main():
    pre_result = run_pooled_decomposition(
        TASK_STATES_DIR / "pre_model" / "best_model_cls.pt",
        condition="pre",
        label="pre_model",
    )
    post_result = run_pooled_decomposition(
        TASK_STATES_DIR / "post_model" / "best_model_cls.pt",
        condition="post",
        label="post_model",
    )

    print_per_roi_table(pre_result)
    print_per_roi_table(post_result)
    print_summary_table([pre_result, post_result])


if __name__ == "__main__":
    main()