"""
================================================================================
LOSO Stability / Perturbation Analysis
================================================================================
Made to perturb a single ROI's diagonal-of-C entries in a held-out subject's 
best_model_cls.pt.

For a single LOSO fold's held-out subject, this script:

    1. Loads that fold's best_model_cls.pt.
    2. Runs the real encoder + control module on the subject's own BOLD
       session to get g_0, C (base), and u_t.
    3. Re-implements the eigenspace recurrence manually (mathematically
       identical to forward()'s parallel_scan + decode -- verified against
       the model's own x_recon as a sanity check at runtime) so that C can
       be varied timestep-by-timestep (forward() itself assumes a single
       fixed C for the whole session).
    4. Builds a step schedule that perturbs ONE chosen ROI (PERTURB_ROI_NAME)
       at a single onset timestep (PULSE_START): unperturbed C_base before
       onset, then a constant additive shift to that ROI's decoded C value
       from onset to the end of the trajectory.
    5. Plots raw BOLD vs. unperturbed reconstruction vs. perturbed
       reconstruction, one panel per ROI, in a 6-row x 4-col grid (matching
       the ncols=4 convention of loso_C_timeline_by_roi.png etc), plus
       supporting plots that isolate the perturbation's effect and its
       K vs. C contribution.

IMPORTANT CAVEAT -- C is diagonal over M=96 MODES, not 24 ROIs:
    There is no literal "this ROI's diagonal entry of C" -- mode space only
    maps to ROI space through the learned decoder W_bar_x. To perturb "ROI
    r" here, we: (1) compute this session's decoder-projected C value at
    every ROI (roi_C = B @ diag(C), the EXACT decode-from-g -- see
    g_decode_matrix() below); (2) build a target vector that is zero
    everywhere except a chosen absolute magnitude (PERT_MAGNITUDE) at
    ROI_IDX; (3) decompose that target back into a mode-space delta via the
    Moore-Penrose pseudoinverse of the decode matrix. Because the decode
    matrix is full row rank (M=96 > N_ROIS=24), this decomposition is EXACT
    -- every other ROI's decoded C value is unchanged by construction, not
    just "small."

    CORRECTION (see g_decode_matrix() docstring): an earlier version of
    this script used A = Re(W_bar_x) directly as the decode matrix. That is
    the decoder for g_bar (post-P_inv, the space the eigenspace recurrence
    actually lives in for READOUT), but C's contribution
    (c_term = P_inv @ (C_t @ u_t)) enters in g-space, BEFORE P_inv is
    applied -- the same space K operates in (K = P @ diag(Lambda) @ P_inv
    acts on g). The mathematically exact decode is therefore
    B = Re(W_bar_x @ P_inv), derived directly from
    x_hat = Re(W_bar_x @ g_bar) = Re(W_bar_x @ P_inv @ g) = Re(B @ g).

Usage:
    python analysis/C_stability_perturbation_analysis.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config import M, N_ROIS, H, T
from preprocessing.load_preprocessed_data import TARGET_ROIS
from analysis.analysis_helper_functions import load_model, compute_K
from training.dataset import BRICKDataset
from training.train import DATA_DIR

# ================================================================================
# CONFIG -- edit these
# ================================================================================
SUBJECT_ID = "sub-fuspd07"       # which LOSO fold / held-out subject
TARGET     = "vim"               # "vim" or "zi"
SESSION    = "pre"               # "pre" or "post"

# Change directory for save
LOSO_DIR = ROOT_DIR / "results" / "training" / "loso_19_fold_beta_0.2_13to1to5_split"
CHECKPOINT_PATH = LOSO_DIR / f"fold_{SUBJECT_ID}" / "best_model_cls.pt"

OUT_DIR = ROOT_DIR / "results" / "stability_perturbation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Perturbation: single step, not a pulse train ---
PERTURB_ROI_NAME = "lh_GPe"                      # which ROI gets perturbed, by name
ROI_IDX = TARGET_ROIS.index(PERTURB_ROI_NAME)    # resolved index into TARGET_ROIS

PULSE_START  = T // 2     # timestep the perturbation switches on (120 of 240)

PERT_MAGNITUDE = 2.0      # absolute additive value in ROI-space (decoder-projected
                          # C units), NOT scaled by std(roi_C) -- tune up/down from here

# Which single ROI to show in the big single-panel overlay plot -- defaults to
# the perturbed ROI itself, but change this (e.g. "lh_cerebellum_motor") to
# zoom into a downstream ROI's response instead.
PLOT_ROI_NAME = PERTURB_ROI_NAME
PLOT_ROI_IDX  = TARGET_ROIS.index(PLOT_ROI_NAME)


CONDITION_TO_SESSION = {"mpre": "pre", "mpost": "post"}


# ================================================================================
# 1. LOAD MODEL + SUBJECT DATA
# ================================================================================
def load_subject_item():
    ds = BRICKDataset(DATA_DIR)
    for i in range(len(ds)):
        item = ds[i]
        if (item["subject_id"] == SUBJECT_ID
                and item["target"] == TARGET
                and CONDITION_TO_SESSION[item["condition_str"]] == SESSION):
            return item
    raise ValueError(
        f"No item found for subject_id={SUBJECT_ID!r}, target={TARGET!r}, "
        f"session={SESSION!r}. Check the ID/target/session against BRICKDataset."
    )


# ================================================================================
# 2. REPLICATE forward()'s ENCODER + CONTROL STEP (front half only)
# ================================================================================
def encode_and_control(model, x):
    """
    Returns (g_0, C_base, u) exactly as BRICK.forward() would compute them
    internally, without running the (fixed-C) parallel scan.
    """
    with torch.no_grad():
        if model.use_ic:
            g_0, _, _ = model.encoder(x)
        else:
            g_0 = torch.zeros(model.m, device=x.device)

        C_base, u, _, _, _ = model.control(x)

    return g_0, C_base, u


# ================================================================================
# 3. MANUAL EIGENSPACE ROLLOUT (allows C to vary per-timestep)
# ================================================================================
def manual_rollout(Lambda, P_inv, g_0, u, W_bar_x, C_seq, return_decomposition=False):
    """
    Re-implements BRICK.forward()'s recurrence:
        g_bar_0 = P_inv @ g_0
        g_bar_t = Lambda * g_bar_{t-1} + P_inv @ (C_t @ u_t),   t = 1..T
        x_hat[t-1] = Re(W_bar_x @ g_bar_t)

    C_seq: either a single (M, M) tensor (constant C, for the sanity check
           against the model's real forward pass) or a list/tuple of T
           (M, M) tensors giving a per-timestep C.

    return_decomposition: if True, also returns (x_K, x_C) -- the decoded
        BOLD signal split into this step's K-propagated-previous-state term
        and this step's fresh C@u_t injection term. x_K + x_C == x_hat
        exactly (decode is linear), so this is a non-approximate split, not
        an estimate. "K term at t" is the memory of all PAST injections
        filtered one more step through K -- not "signal unrelated to C".
    """
    with torch.no_grad():
        Tlen = u.shape[0]
        g0_c = g_0.to(torch.complex64)
        u_c  = u.to(torch.complex64)

        g_bar = P_inv @ g0_c  # eigenspace initial state
        traj, k_terms, c_terms = [], [], []
        for t in range(Tlen):
            C_t = C_seq[t] if isinstance(C_seq, (list, tuple)) else C_seq
            C_t_c = C_t.to(torch.complex64)
            k_term = Lambda * g_bar
            c_term = P_inv @ (C_t_c @ u_c[t])
            g_bar = k_term + c_term
            traj.append(g_bar)
            if return_decomposition:
                k_terms.append(k_term)
                c_terms.append(c_term)

        g_bar_traj = torch.stack(traj, dim=0)          # (T, M) complex
        x_hat = (W_bar_x @ g_bar_traj.T).T.real         # (T, N) real

        if not return_decomposition:
            return x_hat

        k_traj = torch.stack(k_terms, dim=0)
        c_traj = torch.stack(c_terms, dim=0)
        x_K = (W_bar_x @ k_traj.T).T.real
        x_C = (W_bar_x @ c_traj.T).T.real

        check = (x_K + x_C - x_hat).abs().max().item()
        if check > 1e-3:
            print(f"  WARNING: x_K + x_C does not match x_hat (max diff "
                  f"{check:.2e}) -- decomposition is broken, don't trust it.")

    return x_hat, x_K, x_C


# ================================================================================
# 4. ROI <-> MODE DECODE MATRIX
# ================================================================================
def g_decode_matrix(W_bar_x, P_inv):
    """
    B = Re(W_bar_x @ P_inv), shape (N_ROIS, M) real.

    This is the mathematically EXACT decode from g-space to ROI-space.
    Derived directly from the model's own readout: x_hat = Re(W_bar_x @ g_bar),
    and g_bar = P_inv @ g, so x_hat = Re(W_bar_x @ P_inv @ g) = Re(B @ g).

    Why this matters here: C's contribution enters the recurrence as
    c_term = P_inv @ (C_t @ u_t) -- i.e. "C @ u_t" is computed in g-space,
    BEFORE P_inv is applied. So decoding diag(C) to ROI space needs the
    g-space decoder B, not Re(W_bar_x) alone (which decodes g_bar, the
    space AFTER P_inv -- correct for reading out x_hat itself, but one
    step removed from where C actually lives). This also makes B the
    right, shared decoder for the companion K-edge-perturbation analysis,
    since K = P @ diag(Lambda) @ P_inv acts on that same g-space.
    """
    return (W_bar_x @ P_inv).real  # (N_ROIS, M)


def roi_from_modes(A, c):
    """roi_C = A @ c, shape (N_ROIS,) real. c = diag(C), shape (M,) real."""
    return A @ c


def modes_from_roi_target(A, target_roi_vec):
    """
    Given a desired ROI-space perturbation vector (N_ROIS,) -- e.g. zero
    everywhere except a single value at ROI r -- solve for the minimum-norm
    mode-space vector delta_c (M,) such that A @ delta_c == target_roi_vec
    EXACTLY (A is full row rank since M=96 > N_ROIS=24, so this is exact,
    not approximate). Because it's exact, every ROI not named in
    target_roi_vec gets precisely zero perturbation from this delta_c --
    not "small", zero.
    """
    A_pinv = torch.linalg.pinv(A)     # (M, N_ROIS)
    return A_pinv @ target_roi_vec    # (M,)


# ================================================================================
# 5. STEP SCHEDULE (single onset, persists to the end)
# ================================================================================
def build_step_schedule():
    """
    Returns a list of length T: None before PULSE_START, ROI_IDX from
    PULSE_START onward (no "off" -- the perturbation is a permanent step,
    not a pulse).
    """
    return [None] * PULSE_START + [ROI_IDX] * (T - PULSE_START)


# ================================================================================
# 6. BUILD PERTURBED C SEQUENCE (ROI-space target, decomposed to mode-space)
# ================================================================================
def build_C_sequence(C_base, A, schedule, pert_magnitude=PERT_MAGNITUDE):
    """
    Prints this session's baseline decoder-projected C value at every ROI
    (for reference/scale), then builds a per-timestep C sequence where, from
    PULSE_START onward, ROI_IDX's decoded C value is shifted by
    pert_magnitude (an absolute addition, not scaled to the data). The
    target is decomposed back into mode space via modes_from_roi_target()
    and added to C_base's diagonal; the decomposition is exact, so every
    OTHER ROI's decoded C value is left untouched.
    """
    c_base = torch.diagonal(C_base).real if torch.diagonal(C_base).is_complex() \
        else torch.diagonal(C_base)

    roi_C_base = roi_from_modes(A, c_base)                 # (N_ROIS,)
    print(f"\nROI-space C stats for this session (baseline):")
    for r, roi_name in enumerate(TARGET_ROIS):
        marker = "  <-- perturbing this one" if r == ROI_IDX else ""
        print(f"    {roi_name:<30} roi_C = {roi_C_base[r].item():+.4f}{marker}")
    print(f"  injecting an absolute +{pert_magnitude:.4f} onto "
          f"{TARGET_ROIS[ROI_IDX]}'s roi_C from t={PULSE_START} onward")

    target_roi_vec = torch.zeros(N_ROIS, dtype=A.dtype)
    target_roi_vec[ROI_IDX] = pert_magnitude
    delta_c = modes_from_roi_target(A, target_roi_vec)   # (M,) real
    C_perturbed = C_base + torch.diag_embed(delta_c.to(C_base.dtype))

    # --- Diagnostic: does the injection ITSELF land where intended, before
    # u_t or K get involved at all? This is the static decode of delta_c
    # alone through A -- should be ~pert_magnitude at ROI_IDX and ~0
    # everywhere else, confirming modes_from_roi_target()'s pseudoinverse
    # solve is doing what it claims, independent of anything the dynamics
    # do to it afterward. ---
    roi_check = roi_from_modes(A, delta_c)
    print(f"\nStatic decode of the injected delta_c alone (A @ delta_c) -- "
          f"should be ~{pert_magnitude:.2f} at {TARGET_ROIS[ROI_IDX]} and ~0 elsewhere:")
    for r, roi_name in enumerate(TARGET_ROIS):
        marker = "  <-- target" if r == ROI_IDX else ""
        print(f"    {roi_name:<30} A@delta_c = {roi_check[r].item():+.4f}{marker}")

    C_seq = [C_base if roi is None else C_perturbed for roi in schedule]
    return C_seq, roi_C_base


# ================================================================================
# 7. PLOT
# ================================================================================
def plot_grid(x_raw, x_hat_base, x_hat_pert, out_path):
    rois = list(TARGET_ROIS)
    ncols = 4
    nrows = int(np.ceil(len(rois) / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3 * nrows), squeeze=False)
    t_axis = np.arange(T)

    for idx, roi in enumerate(rois):
        ax = axes[idx // ncols][idx % ncols]

        ax.plot(t_axis, x_raw[:, idx], color="black", lw=1.0, label="raw BOLD")
        ax.plot(t_axis, x_hat_base[:, idx], color="#4C72B0", lw=1.2,
                linestyle="--", label="predicted (unperturbed)")
        ax.plot(t_axis, x_hat_pert[:, idx], color="#DD8452", lw=1.2,
                label="predicted (perturbed)")

        ax.axvline(PULSE_START, color="red", lw=1.2, linestyle=":")
        if idx == ROI_IDX:
            ax.axvspan(PULSE_START, T, color="red", alpha=0.12)

        ax.set_title(roi, fontsize=9)
        ax.tick_params(labelsize=7)
        if idx % ncols == 0:
            ax.set_ylabel("BOLD (z-scored)", fontsize=8)
        if idx // ncols == nrows - 1:
            ax.set_xlabel("timepoint", fontsize=8)

    for idx in range(len(rois), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, 1.03))
    fig.suptitle(
        f"Stability perturbation -- {SUBJECT_ID}, {TARGET.upper()} {SESSION}\n"
        f"red dotted line = perturbation onset (t={PULSE_START}); "
        f"red band = {TARGET_ROIS[ROI_IDX]} held perturbed from onset to the end",
        fontsize=12, y=1.07,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


def plot_delta_grid(delta, out_path):
    """
    Isolated perturbation effect: x_hat_pert - x_hat_base, per ROI. Because
    the recurrence is linear given a fixed C_t sequence, this difference is
    EXACTLY the perturbation's own contribution to every ROI's trajectory --
    no baseline BOLD signal or subject-specific noise riding along with it.
    This is the honest way to see how far/fast the perturbation spreads,
    since the raw/predicted overlay plot conflates "spread" with the
    session's own dynamics.
    """
    rois = list(TARGET_ROIS)
    ncols = 4
    nrows = int(np.ceil(len(rois) / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3 * nrows), squeeze=False)
    t_axis = np.arange(T)
    ymax = np.abs(delta).max() * 1.1

    for idx, roi in enumerate(rois):
        ax = axes[idx // ncols][idx % ncols]
        color = "#DD8452" if idx == ROI_IDX else "#4C72B0"
        ax.plot(t_axis, delta[:, idx], color=color, lw=1.2)
        ax.axhline(0, color="black", lw=0.5)
        ax.axvline(PULSE_START, color="red", lw=1.0, linestyle=":")
        ax.set_ylim(-ymax, ymax)
        ax.set_title(roi, fontsize=9)
        ax.tick_params(labelsize=7)
        if idx % ncols == 0:
            ax.set_ylabel("Δx_hat (perturbed - unperturbed)", fontsize=8)
        if idx // ncols == nrows - 1:
            ax.set_xlabel("timepoint", fontsize=8)

    for idx in range(len(rois), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.suptitle(
        f"Isolated perturbation effect on reconstruction -- {SUBJECT_ID}, "
        f"{TARGET.upper()} {SESSION}\n"
        f"{TARGET_ROIS[ROI_IDX]} perturbed from t={PULSE_START} (orange panel); "
        f"all other panels show pure spread through K, decoupled from the "
        f"session's own BOLD signal",
        fontsize=12, y=1.05,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


def plot_k_c_decomposition_grid(x_K, x_C, out_path):
    """
    24-panel grid, one per ROI: signed stacked area showing how much of the
    reconstructed BOLD at each timepoint comes from K (purple, this step's
    propagation of everything the system already remembers) vs. C (green,
    this step's fresh control injection). x_K + x_C == x_hat exactly.

    Stacking convention: purple fills [0, x_K]; green fills [x_K, x_K+x_C].
    Where C is the same sign as K, green extends the total further from 0
    (reinforcing). Where C is the opposite sign, green pulls the green edge
    back toward (or past) zero, visibly eating into the purple span
    (undercutting) -- the stacking itself shows this, no extra annotation
    needed, but the total-height line makes the net effect explicit too.
    """
    rois = list(TARGET_ROIS)
    ncols = 4
    nrows = int(np.ceil(len(rois) / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3 * nrows), squeeze=False)
    t_axis = np.arange(T)
    total = x_K + x_C
    ymax = max(np.abs(x_K).max(), np.abs(total).max()) * 1.1

    for idx, roi in enumerate(rois):
        ax = axes[idx // ncols][idx % ncols]
        k = x_K[:, idx]
        c = x_C[:, idx]
        stack_top = k + c

        ax.fill_between(t_axis, 0, k, color="#6A3D9A", alpha=0.6, label="K (propagated memory)")
        ax.fill_between(t_axis, k, stack_top, color="#33A02C", alpha=0.6, label="C (fresh injection)")
        ax.plot(t_axis, stack_top, color="black", lw=0.8, label="total (x_hat)")
        ax.axhline(0, color="black", lw=0.4)
        ax.axvline(PULSE_START, color="red", lw=1.0, linestyle=":")

        ax.set_ylim(-ymax, ymax)
        ax.set_title(roi, fontsize=9)
        ax.tick_params(labelsize=7)
        if idx % ncols == 0:
            ax.set_ylabel("BOLD contribution", fontsize=8)
        if idx // ncols == nrows - 1:
            ax.set_xlabel("timepoint", fontsize=8)

    for idx in range(len(rois), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, 1.03))
    fig.suptitle(
        f"K vs. C contribution to reconstructed BOLD -- {SUBJECT_ID}, "
        f"{TARGET.upper()} {SESSION} (perturbed run)\n"
        f"K = this step's propagation of everything already remembered "
        f"(NOT \"signal unrelated to C\" -- it's the accumulated memory of "
        f"all past C injections); C = this step's fresh injection only",
        fontsize=11, y=1.08,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


def plot_single_roi_overlay(x_raw, x_hat_base, x_hat_pert, roi_idx, out_path):
    """
    Same three traces as plot_grid (raw BOLD, unperturbed prediction,
    perturbed prediction), but a single, larger panel for one ROI instead
    of the crowded 24-panel grid -- for zooming into a specific region of
    interest.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    t_axis = np.arange(T)
    roi_name = TARGET_ROIS[roi_idx]

    ax.plot(t_axis, x_raw[:, roi_idx], color="black", lw=1.3, label="raw BOLD")
    ax.plot(t_axis, x_hat_base[:, roi_idx], color="#4C72B0", lw=1.6,
            linestyle="--", label="predicted (unperturbed)")
    ax.plot(t_axis, x_hat_pert[:, roi_idx], color="#DD8452", lw=1.6,
            label="predicted (perturbed)")

    ax.axvline(PULSE_START, color="red", lw=1.4, linestyle=":", label=f"onset (t={PULSE_START})")
    if roi_idx == ROI_IDX:
        ax.axvspan(PULSE_START, T, color="red", alpha=0.10,
                   label=f"{PERTURB_ROI_NAME} held perturbed")

    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("timepoint")
    ax.set_ylabel("BOLD (z-scored)")
    ax.set_title(
        f"{roi_name} -- {SUBJECT_ID}, {TARGET.upper()} {SESSION}\n"
        f"perturbed ROI: {PERTURB_ROI_NAME}, magnitude={PERT_MAGNITUDE}",
        fontsize=11,
    )
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


def print_impact_ranking(delta):
    """
    Ranks every ROI by how hard the perturbation hits it (peak |Δx_hat|
    after onset) and how fast (first timestep the deviation exceeds 10% of
    that ROI's own peak). Confirms causality as a side effect: pre-onset
    deltas should be ~0 for every ROI.
    """
    pre_onset_max = np.abs(delta[:PULSE_START]).max()
    print(f"\nCausality check -- max |Δx_hat| BEFORE onset (should be ~0): "
          f"{pre_onset_max:.2e}")

    post = delta[PULSE_START:]
    rows = []
    for r, roi_name in enumerate(TARGET_ROIS):
        col = post[:, r]
        peak = np.abs(col).max()
        thresh = 0.1 * peak if peak > 0 else np.inf
        onset_idx = np.argmax(np.abs(col) >= thresh) if peak > 0 else -1
        rows.append((roi_name, peak, onset_idx))

    rows.sort(key=lambda r: -r[1])
    print(f"\nPer-ROI impact ranking (peak |Δx_hat| after onset, "
          f"latency = timesteps after onset to reach 10% of that ROI's own peak):")
    for roi_name, peak, onset_idx in rows:
        marker = "  <-- directly perturbed" if roi_name == TARGET_ROIS[ROI_IDX] else ""
        print(f"    {roi_name:<30} peak={peak:6.3f}   latency={onset_idx:4d}{marker}")


def variance_decomposition(x_K, x_C):
    """
    Well-posed version of "% signal from C vs K": splits Var(x_hat) into an
    instantaneous-injection contribution (x_C = Cu_t, the s=t term) and an
    accumulated/propagated-history contribution (x_K = this step's K-
    propagation of g_bar_{t-1}, which by induction already equals
    Sum_{s<t} K^(t-s) C u_s -- i.e. everything the s<t terms produced).

    x_K and x_C are correlated in general, so Var(x_K)+Var(x_C) != Var(total)
    -- there's a shared covariance term. Uses the standard exact fair split:
    each term gets its own variance plus half the shared covariance, which
    sums to the total variance exactly (no unexplained residual):
        contribution_K = Var(x_K) + Cov(x_K, x_C)
        contribution_C = Var(x_C) + Cov(x_K, x_C)
        contribution_K + contribution_C == Var(x_K + x_C) exactly.

    Returns a list of per-ROI dicts.
    """
    Tlen, Nrois = x_K.shape
    rows = []
    for r in range(Nrois):
        k, c = x_K[:, r], x_C[:, r]
        var_k, var_c = np.var(k), np.var(c)
        cov_kc = np.cov(k, c)[0, 1]
        var_total = np.var(k + c)

        contrib_k = var_k + cov_kc
        contrib_c = var_c + cov_kc
        pct_k = 100 * contrib_k / var_total if var_total > 0 else np.nan
        pct_c = 100 * contrib_c / var_total if var_total > 0 else np.nan

        rows.append({
            "roi": TARGET_ROIS[r], "var_K": var_k, "var_C": var_c,
            "cov_KC": cov_kc, "var_total": var_total,
            "pct_accumulated_K": pct_k, "pct_instantaneous_C": pct_c,
        })
    return rows


def print_variance_decomposition(rows, label):
    print(f"\n% signal from instantaneous C-injection vs. accumulated K-history "
          f"-- {label}:")
    print(f"  {'ROI':<30} {'% instant. (C)':>15} {'% accum. (K)':>14} {'Var(total)':>12}")
    for r in rows:
        print(f"  {r['roi']:<30} {r['pct_instantaneous_C']:>14.1f}% "
              f"{r['pct_accumulated_K']:>13.1f}% {r['var_total']:>12.4f}")

    pooled_var_total = sum(r["var_total"] for r in rows)
    pooled_pct_c = sum(r["pct_instantaneous_C"] * r["var_total"] for r in rows) / pooled_var_total
    pooled_pct_k = sum(r["pct_accumulated_K"] * r["var_total"] for r in rows) / pooled_var_total
    print(f"  {'VARIANCE-WEIGHTED MEAN':<30} {pooled_pct_c:>14.1f}% {pooled_pct_k:>13.1f}%")


def print_scale_diagnostics(C_base, Lambda):
    """
    Secondary diagnostic: raw magnitude comparison of C vs K. Flagged (per
    your own caveat) as not obviously meaningful on its own -- K's
    eigenvalues are dimensionless decay factors constrained <1 for
    stability, C is a gain on a stochastic input -- different units, so
    this is a rough scale check, not the primary answer (that's
    variance_decomposition() above).
    """
    c_diag = torch.diagonal(C_base)
    c_diag = c_diag.real if c_diag.is_complex() else c_diag
    c_frob = c_diag.norm().item()          # Frobenius norm of diag(C) (C is diagonal)
    c_std  = c_diag.std().item()

    lambda_abs = Lambda.abs()
    spectral_radius = lambda_abs.max().item()
    lambda_mean_abs = lambda_abs.mean().item()

    print(f"\nScale diagnostic (C vs K magnitudes -- different units, see caveat above):")
    print(f"  ||diag(C)||_F = {c_frob:.4f}   std(diag(C)) = {c_std:.4f}")
    print(f"  spectral radius of K (max|Lambda|) = {spectral_radius:.4f}   "
          f"mean|Lambda| = {lambda_mean_abs:.4f}")
    print(f"  (K's eigenvalues should be <1 for a stable system; if "
          f"spectral radius is >=1, that's worth flagging on its own)")


def to_complex_tensor(x):
    """compute_K() numpy-ifies its outputs (for the plotting/stats callers
    elsewhere in the codebase) -- cast back to a complex64 torch tensor."""
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(torch.complex64)
    if torch.is_tensor(x):
        return x.to(torch.complex64)
    return torch.as_tensor(x, dtype=torch.complex64)


# ================================================================================
# MAIN
# ================================================================================
def main():
    print(f"Loading fold checkpoint: {CHECKPOINT_PATH}")
    model = load_model(CHECKPOINT_PATH)
    K, Lambda, W_bar_x = compute_K(model)
    Lambda  = to_complex_tensor(Lambda)
    W_bar_x = to_complex_tensor(W_bar_x)
    P_inv = model.P_inv

    item = load_subject_item()
    x = item["x"]
    print(f"Loaded {SUBJECT_ID} / {TARGET} / {SESSION}, x shape = {tuple(x.shape)}")

    g_0, C_base, u = encode_and_control(model, x)

    # --- Sanity check: manual rollout with constant C_base must match the
    # model's own forward() reconstruction (same math, different code path) ---
    with torch.no_grad():
        real_out = model(x, item["lifus_condition"],
                          kl_g0_weight=1.0, kl_u_weight=1.0, apply_free_bits=False)
    x_hat_base, x_K_base, x_C_base = manual_rollout(
        Lambda, P_inv, g_0, u, W_bar_x, C_seq=C_base, return_decomposition=True
    )
    max_diff = (x_hat_base - real_out["x_recon"]).abs().max().item()
    print(f"Sanity check -- max |manual_rollout - forward().x_recon| = {max_diff:.2e}")
    if max_diff > 1e-3:
        print("  WARNING: manual rollout does not match forward() closely -- "
              "check _assemble_u_bar/parallel_scan indexing before trusting "
              "the perturbation results below.")

    # --- ROI <-> mode decode matrix (exact decode-from-g, see g_decode_matrix()) ---
    A = g_decode_matrix(W_bar_x, P_inv)

    # --- Step schedule + perturbed C sequence ---
    schedule = build_step_schedule()
    C_seq, roi_C_base = build_C_sequence(C_base, A, schedule)

    x_hat_pert, x_K, x_C = manual_rollout(
        Lambda, P_inv, g_0, u, W_bar_x, C_seq=C_seq, return_decomposition=True
    )

    effect = (x_hat_pert - x_hat_base).abs()
    print(f"\nEffect size on reconstructed BOLD: "
          f"max |Δx_hat| = {effect.max().item():.4f}, "
          f"mean |Δx_hat| = {effect.mean().item():.4f} "
          f"(raw BOLD is z-scored, so these are in std-dev units -- "
          f"if max is still << 1, bump PERT_MAGNITUDE further)")

    delta_np = (x_hat_pert - x_hat_base).numpy()
    print_impact_ranking(delta_np)

    # --- % signal from instantaneous C-injection vs. accumulated K-history ---
    print_scale_diagnostics(C_base, Lambda)
    rows_base = variance_decomposition(x_K_base.numpy(), x_C_base.numpy())
    print_variance_decomposition(rows_base, label=f"baseline (unperturbed) session")
    rows_pert = variance_decomposition(x_K.numpy(), x_C.numpy())
    print_variance_decomposition(rows_pert, label=f"perturbed run ({PERTURB_ROI_NAME} stepped at t={PULSE_START})")

    # --- Plot: raw / predicted / perturbed overlay ---
    out_path = OUT_DIR / f"stability_perturbation_{SUBJECT_ID}_{TARGET}_{SESSION}.png"
    plot_grid(
        x_raw=x.numpy(),
        x_hat_base=x_hat_base.numpy(),
        x_hat_pert=x_hat_pert.numpy(),
        out_path=out_path,
    )

    # --- Plot: isolated perturbation effect (perturbed - unperturbed), no
    # baseline BOLD signal riding along with it -- the clean "spread" view ---
    delta_out_path = OUT_DIR / f"stability_perturbation_delta_{SUBJECT_ID}_{TARGET}_{SESSION}.png"
    plot_delta_grid(delta_np, delta_out_path)

    # --- Plot: K vs C contribution to the perturbed run's reconstruction ---
    kc_out_path = OUT_DIR / f"stability_perturbation_K_vs_C_{SUBJECT_ID}_{TARGET}_{SESSION}.png"
    plot_k_c_decomposition_grid(x_K.numpy(), x_C.numpy(), kc_out_path)

    # --- Plot: single-ROI zoom (defaults to the perturbed ROI itself --
    # change PLOT_ROI_NAME at the top to inspect a different ROI instead) ---
    single_out_path = OUT_DIR / f"stability_perturbation_single_{PLOT_ROI_NAME}_{SUBJECT_ID}_{TARGET}_{SESSION}.png"
    plot_single_roi_overlay(
        x_raw=x.numpy(),
        x_hat_base=x_hat_base.numpy(),
        x_hat_pert=x_hat_pert.numpy(),
        roi_idx=PLOT_ROI_IDX,
        out_path=single_out_path,
    )


if __name__ == "__main__":
    main()