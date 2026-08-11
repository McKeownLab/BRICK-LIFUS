"""
================================================================================
LOSO K Edge Perturbation Analysis
================================================================================

Sensitivity/robustness proof-of-concept for the shared Koopman operator K:
strengthen ONE directed ROI-pair edge in K's decoded ROI x ROI heatmap
(source ROI -> target ROI), re-run the session's trajectory, and see whether
the effect stays localized to the target ROI or spreads through the network.

Companion to the C_stability_perturbation_analysis.py (the C-perturbation
script). Differences as follows:

  - C is a fresh, ADDITIVE injection each timestep (C_t @ u_t). Perturbing
    it adds a bounded kick; the effect on x_hat is exactly separable via
    x_hat_pert - x_hat_base with no risk of blowing up.
  - K is MULTIPLICATIVE and PERSISTENT: in eigenspace, g_bar_t = Lambda *
    g_bar_{t-1}. Perturbing it changes how everything the system already
    remembers keeps propagating, every single step from onset onward. If
    the perturbed operator's eigenvalues have modulus >= 1 anywhere, the
    trajectory diverges -- not a finding about connectivity, a numerical
    artifact. See clamp_for_stability() below for how this is handled.

WHERE K LIVES, AND WHY THE DECODE MATCHES C's:
    compute_K() returns K = P @ diag(Lambda) @ P_inv, which acts on g (the
    SAME pre-P_inv space that C's contribution enters through:
    c_term = P_inv @ (C_t @ u_t)). So the exact decode-from-g used here,
    B = Re(W_bar_x @ P_inv), is identical in derivation to the one now used
    in the C script -- see g_decode_matrix() below..

DECODING K TO ROI SPACE, AND WHY THE SINGLE-EDGE PERTURBATION IS EXACT:
    K_roi = B @ K_g @ B_pinv, a two-sided generalization of the C script's
    one-sided vector decode. Because B is full row rank (M=96 > N_ROIS=24),
    B @ B_pinv = I_N exactly, so setting
        delta_K_g = B_pinv @ delta_K_roi @ B
    guarantees B @ delta_K_g @ B_pinv == delta_K_roi EXACTLY. If delta_K_roi
    is zero everywhere except one target cell, every OTHER cell of the
    decoded K_roi heatmap is left at precisely zero perturbation.

SOURCE ROI SELECTION:
    select_source_roi() picks whichever ROI has the strongest baseline 
    |K_roi[target, source]| edge, feeding into the target ROI (self-edge 
    excluded), and prints the  incoming row so the choice is auditable. 
    Override via SOURCE_ROI_NAME_OVERRIDE if you want a specific pathway 
    instead.

STABILITY, HANDLED BY CONSTRUCTION RATHER THAN AFTER THE FACT:
    Clamp_for_stability() bisection-searches for the largest
    scale <= 1.0 (fraction of the REQUESTED edge magnitude) that keeps the
    perturbed operator's spectral radius under STABILITY_MARGIN, and uses
    that scale from the start. If the full requested magnitude is already
    stable (the common case), nothing is scaled and this is silent besides
    a confirmation line.

Usage:
    python analysis/K_stability_perturbation_analysis.py
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
SUBJECT_ID = "sub-fuspd07"
TARGET     = "vim"
SESSION    = "pre"

LOSO_DIR = ROOT_DIR / "results" / "training" / "loso_19_fold_beta_0.2_13to1to5_split"
CHECKPOINT_PATH = LOSO_DIR / f"fold_{SUBJECT_ID}" / "best_model_cls.pt"

OUT_DIR = ROOT_DIR / "results" / "K_stability_perturbation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Edge being strengthened: SOURCE_ROI -> TARGET_ROI_NAME ---
TARGET_ROI_NAME = "lh_GPe"                          # edge destination
TARGET_ROI_IDX  = TARGET_ROIS.index(TARGET_ROI_NAME)

# Source is auto-selected by select_source_roi() (largest baseline |edge|
# into the target, self-edge excluded). Set this to an ROI name to force a
# specific source instead of the automatic choice.
SOURCE_ROI_NAME_OVERRIDE = None

PULSE_START = T // 2       # onset timestep; perturbation holds to the end (no "off"),
                            # same convention as the C script

EDGE_MAGNITUDE = 2.0       # REQUESTED absolute shift in K_roi[target, source].
                            # May be automatically scaled down to stay stable --
                            # see clamp_for_stability(); actual applied magnitude
                            # is printed at runtime.

STABILITY_MARGIN = 0.999   # max allowed spectral radius of the perturbed
                            # propagation operator (must stay < 1 for a stable system)

CONDITION_TO_SESSION = {"mpre": "pre", "mpost": "post"}


# ================================================================================
# 1. LOAD MODEL + SUBJECT DATA  (mirrors C_stability_perturbation_analysis.py)
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


def encode_and_control(model, x):
    """Returns (g_0, C_base, u) exactly as BRICK.forward() would compute them."""
    with torch.no_grad():
        if model.use_ic:
            g_0, _, _ = model.encoder(x)
        else:
            g_0 = torch.zeros(model.m, device=x.device)
        C_base, u, _, _, _ = model.control(x)
    return g_0, C_base, u


def to_complex_tensor(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(torch.complex64)
    if torch.is_tensor(x):
        return x.to(torch.complex64)
    return torch.as_tensor(x, dtype=torch.complex64)


# ================================================================================
# 2. DECODE MATRIX -- shared convention with the (corrected) C script
# ================================================================================
def g_decode_matrix(W_bar_x, P_inv):
    """
    B = Re(W_bar_x @ P_inv), shape (N_ROIS, M) real. Exact decode from
    g-space (where both K and C act) to ROI-space. See module docstring.
    """
    return (W_bar_x @ P_inv).real


# ================================================================================
# 3. DECODE K TO ROI x ROI SPACE, AND SELECT THE SOURCE ROI
# ================================================================================
def decode_K_to_roi(K_g, B):
    """
    K_roi = B @ K_g @ B_pinv, shape (N_ROIS, N_ROIS) real (imaginary part
    dropped, same convention as x_hat = Re(...) -- K_g's complex conjugate
    eigenmode pairs combine to real dynamics on real inputs).

    Two-sided generalization of the C script's one-sided A @ c decode:
    B @ B_pinv = I_N exactly (B is full row rank, M=96 > N_ROIS=24), so this
    decode is the natural, non-approximate extension to a full operator.

    Returns (K_roi, B_pinv) -- B_pinv is reused by build_delta_K_g().
    """
    B_pinv = torch.linalg.pinv(B)          # (M, N_ROIS)
    K_roi = (B @ K_g.to(B.dtype) @ B_pinv).real
    return K_roi, B_pinv


def select_source_roi(K_roi, target_idx, override_name=None):
    """
    Auto-selects the source ROI with the largest-magnitude baseline directed
    edge INTO target_idx (self-edge excluded): argmax_i |K_roi[target_idx, i]|
    for i != target_idx. Prints the full incoming row so the pick is
    auditable rather than asserted. Pass override_name to force a specific
    source instead.
    """
    if override_name is not None:
        print(f"\nSource ROI forced via override: {override_name}")
        return TARGET_ROIS.index(override_name)

    row = K_roi[target_idx, :].clone()
    row[target_idx] = 0.0  # exclude self-edge from consideration

    order = torch.argsort(row.abs(), descending=True)
    print(f"\nBaseline decoded K_roi row feeding INTO {TARGET_ROIS[target_idx]} "
          f"(K_roi[target, source] for every candidate source, self-edge excluded, "
          f"sorted by |magnitude|):")
    for rank, i in enumerate(order.tolist()):
        marker = "  <-- selected (largest magnitude)" if rank == 0 else ""
        print(f"    {TARGET_ROIS[i]:<30} K_roi[target,source] = {row[i].item():+.4f}{marker}")

    return order[0].item()


# ================================================================================
# 4. BUILD THE SINGLE-EDGE PERTURBATION (exact, two-sided pseudoinverse)
# ================================================================================
def build_delta_K_g(B, B_pinv, target_idx, source_idx, magnitude):
    """
    delta_K_roi: (N_ROIS, N_ROIS), zero everywhere except `magnitude` at
    [target_idx, source_idx].

    delta_K_g = B_pinv @ delta_K_roi @ B   (M, M)

    Exactness: B @ delta_K_g @ B_pinv
             = B @ B_pinv @ delta_K_roi @ B @ B_pinv
             = I_N @ delta_K_roi @ I_N            (B @ B_pinv = I_N exactly)
             = delta_K_roi
    So decoding this delta back through B on both sides reproduces the
    single-cell target exactly -- every other cell of K_roi is untouched by
    construction, not approximately.
    """
    delta_K_roi = torch.zeros(N_ROIS, N_ROIS, dtype=B.dtype)
    delta_K_roi[target_idx, source_idx] = magnitude
    delta_K_g = (B_pinv.to(torch.complex64) @ delta_K_roi.to(torch.complex64)
                 @ B.to(torch.complex64))
    return delta_K_g, delta_K_roi


# ================================================================================
# 5. STABILITY CLAMP
# ================================================================================
def spectral_radius(mat):
    """Max |eigenvalue| of a (possibly non-normal) complex matrix."""
    return torch.linalg.eigvals(mat).abs().max().item()


def clamp_for_stability(Lambda, P, P_inv, delta_K_g, margin=STABILITY_MARGIN,
                         max_iters=40):
    """
    The perturbed propagation operator in g_bar-space is
        K_bar_eff = diag(Lambda) + scale * (P_inv @ delta_K_g @ P)
    (derived from g_bar_t = P_inv @ g_t = P_inv @ (K + scale*delta_K_g) @ g_{t-1}
    = P_inv @ K @ P @ g_bar_{t-1} + scale * P_inv @ delta_K_g @ P @ g_bar_{t-1},
    and P_inv @ K @ P = diag(Lambda) by construction).

    Bisection-searches scale in [0, 1] (fraction of the FULL requested
    EDGE_MAGNITUDE) for the largest value keeping spectral_radius(K_bar_eff)
    <= margin. Returns (scale, K_bar_delta) where K_bar_delta is the
    g_bar-space delta already multiplied by the chosen scale, ready to add
    to diag(Lambda) directly in the rollout.
    """
    K_bar_delta_full = P_inv @ delta_K_g @ P

    def radius_at(scale):
        return spectral_radius(torch.diag(Lambda) + scale * K_bar_delta_full)

    base_radius = radius_at(0.0)
    print(f"\nStability check -- baseline spectral radius (unperturbed K): "
          f"{base_radius:.6f}")
    if base_radius >= margin:
        print("  WARNING: baseline K is already at/above the stability margin. "
              "This is a pre-existing property of this fold's fitted K, not "
              "something the perturbation caused -- worth flagging on its own.")

    full_radius = radius_at(1.0)
    print(f"Requested edge magnitude ({EDGE_MAGNITUDE}) -> spectral radius = "
          f"{full_radius:.6f} (margin = {margin})")

    if full_radius <= margin:
        print("  Stable at full requested magnitude -- no scaling needed.")
        return 1.0, K_bar_delta_full

    lo, hi = 0.0, 1.0
    for _ in range(max_iters):
        mid = (lo + hi) / 2
        if radius_at(mid) <= margin:
            lo = mid
        else:
            hi = mid

    print(f"  UNSTABLE at full requested magnitude -- auto-scaled down to "
          f"{lo:.4f}x ({lo * EDGE_MAGNITUDE:.4f} absolute) to keep spectral "
          f"radius <= {margin} (resulting radius = {radius_at(lo):.6f}). "
          f"This only shrinks the one perturbed cell uniformly -- it does not "
          f"reintroduce leakage into any other cell of K_roi.")
    return lo, lo * K_bar_delta_full


# ================================================================================
# 6. ROLLOUT -- full-matrix K_bar_eff instead of elementwise Lambda * g_bar
# ================================================================================
def manual_rollout_K(Lambda, g_0, C_base, u, W_bar_x, P_inv, K_bar_delta, schedule):
    """
    Generalizes the C script's manual_rollout: perturbing K breaks the
    diagonal structure in eigenspace, so the elementwise `Lambda * g_bar`
    step becomes a full matmul `K_bar_eff @ g_bar`.

        g_bar_0 = P_inv @ g_0
        g_bar_t = K_bar_eff_t @ g_bar_{t-1} + P_inv @ (C_base @ u_t)
        x_hat[t-1] = Re(W_bar_x @ g_bar_t)

    K_bar_eff_t is diag(Lambda) before PULSE_START, and
    diag(Lambda) + K_bar_delta from PULSE_START onward (schedule[t] True).
    C is held at the real fitted C_base throughout -- unperturbed -- so this
    analysis isolates K's contribution only, same "vary one thing" logic as
    the C script isolating C while holding K fixed.
    """
    with torch.no_grad():
        Tlen = u.shape[0]
        g0_c = g_0.to(torch.complex64)
        u_c = u.to(torch.complex64)
        C_base_c = C_base.to(torch.complex64)
        Lambda_diag = torch.diag(Lambda)

        g_bar = P_inv @ g0_c
        traj = []
        for t in range(Tlen):
            K_bar_eff = (Lambda_diag + K_bar_delta) if schedule[t] else Lambda_diag
            c_term = P_inv @ (C_base_c @ u_c[t])
            g_bar = K_bar_eff @ g_bar + c_term
            traj.append(g_bar)

        g_bar_traj = torch.stack(traj, dim=0)
        x_hat = (W_bar_x @ g_bar_traj.T).T.real
        return x_hat


def build_step_schedule():
    """False before PULSE_START, True from PULSE_START onward (held, no 'off')."""
    return [False] * PULSE_START + [True] * (T - PULSE_START)


# ================================================================================
# 7. PLOTS -- same style as loso_stability_perturbation.py
# ================================================================================
def plot_grid(x_raw, x_hat_base, x_hat_pert, source_roi_idx, out_path):
    rois = list(TARGET_ROIS)
    ncols = 4
    nrows = int(np.ceil(len(rois) / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3 * nrows), squeeze=False)
    t_axis = np.arange(T)

    for idx, roi in enumerate(rois):
        ax = axes[idx // ncols][idx % ncols]

        ax.plot(t_axis, x_raw[:, idx], color="black", lw=1.0, label="raw BOLD")
        ax.plot(t_axis, x_hat_base[:, idx], color="#4C72B0", lw=1.2,
                linestyle="--", label="predicted (baseline K)")
        ax.plot(t_axis, x_hat_pert[:, idx], color="#DD8452", lw=1.2,
                label="predicted (edge-perturbed K)")

        ax.axvline(PULSE_START, color="red", lw=1.2, linestyle=":")
        if idx == TARGET_ROI_IDX:
            ax.axvspan(PULSE_START, T, color="red", alpha=0.12)
        if idx == source_roi_idx:
            ax.axvspan(PULSE_START, T, color="#33A02C", alpha=0.08)

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
        f"K edge perturbation -- {SUBJECT_ID}, {TARGET.upper()} {SESSION}\n"
        f"edge strengthened: {TARGET_ROIS[source_roi_idx]} -> {TARGET_ROI_NAME} "
        f"from t={PULSE_START} onward; red band = target ROI, green band = source ROI",
        fontsize=12, y=1.07,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


def plot_delta_grid(delta, source_roi_idx, out_path):
    """Isolated perturbation effect: x_hat_pert - x_hat_base, per ROI."""
    rois = list(TARGET_ROIS)
    ncols = 4
    nrows = int(np.ceil(len(rois) / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3 * nrows), squeeze=False)
    t_axis = np.arange(T)
    ymax = np.abs(delta).max() * 1.1

    for idx, roi in enumerate(rois):
        ax = axes[idx // ncols][idx % ncols]
        if idx == TARGET_ROI_IDX:
            color = "#DD8452"
        elif idx == source_roi_idx:
            color = "#33A02C"
        else:
            color = "#4C72B0"
        ax.plot(t_axis, delta[:, idx], color=color, lw=1.2)
        ax.axhline(0, color="black", lw=0.5)
        ax.axvline(PULSE_START, color="red", lw=1.0, linestyle=":")
        ax.set_ylim(-ymax, ymax)
        ax.set_title(roi, fontsize=9)
        ax.tick_params(labelsize=7)
        if idx % ncols == 0:
            ax.set_ylabel("Δx_hat (perturbed - baseline)", fontsize=8)
        if idx // ncols == nrows - 1:
            ax.set_xlabel("timepoint", fontsize=8)

    for idx in range(len(rois), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.suptitle(
        f"Isolated K edge-perturbation effect -- {SUBJECT_ID}, {TARGET.upper()} {SESSION}\n"
        f"{TARGET_ROIS[source_roi_idx]} -> {TARGET_ROI_NAME} edge strengthened "
        f"from t={PULSE_START} (orange=target, green=source); all other panels show "
        f"pure spread through K, decoupled from the session's own BOLD signal",
        fontsize=11, y=1.05,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


def plot_single_roi_overlay(x_raw, x_hat_base, x_hat_pert, roi_idx, source_roi_idx, out_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    t_axis = np.arange(T)
    roi_name = TARGET_ROIS[roi_idx]

    ax.plot(t_axis, x_raw[:, roi_idx], color="black", lw=1.3, label="raw BOLD")
    ax.plot(t_axis, x_hat_base[:, roi_idx], color="#4C72B0", lw=1.6,
            linestyle="--", label="predicted (baseline K)")
    ax.plot(t_axis, x_hat_pert[:, roi_idx], color="#DD8452", lw=1.6,
            label="predicted (edge-perturbed K)")

    ax.axvline(PULSE_START, color="red", lw=1.4, linestyle=":", label=f"onset (t={PULSE_START})")
    if roi_idx == TARGET_ROI_IDX:
        ax.axvspan(PULSE_START, T, color="red", alpha=0.10, label="target ROI, held perturbed")

    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("timepoint")
    ax.set_ylabel("BOLD (z-scored)")
    ax.set_title(
        f"{roi_name} -- {SUBJECT_ID}, {TARGET.upper()} {SESSION}\n"
        f"edge {TARGET_ROIS[source_roi_idx]} -> {TARGET_ROI_NAME}, "
        f"requested magnitude={EDGE_MAGNITUDE}",
        fontsize=11,
    )
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


def print_impact_ranking(delta, source_roi_idx):
    """
    Same logic as the C script: ranks every ROI by peak |Δx_hat| after
    onset and latency to 10% of that peak. Pre-onset deltas should be ~0
    for every ROI (causality check) -- if not, something upstream is wrong.
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
        marker = ""
        if roi_name == TARGET_ROI_NAME:
            marker = "  <-- edge target"
        elif roi_name == TARGET_ROIS[source_roi_idx]:
            marker = "  <-- edge source"
        print(f"    {roi_name:<30} peak={peak:6.3f}   latency={onset_idx:4d}{marker}")


# ================================================================================
# MAIN
# ================================================================================
def main():
    print(f"Loading fold checkpoint: {CHECKPOINT_PATH}")
    model = load_model(CHECKPOINT_PATH)
    K_g, Lambda, W_bar_x = compute_K(model)
    K_g     = to_complex_tensor(K_g)
    Lambda  = to_complex_tensor(Lambda)
    W_bar_x = to_complex_tensor(W_bar_x)
    P_inv = model.P_inv
    P = torch.linalg.inv(P_inv)

    item = load_subject_item()
    x = item["x"]
    print(f"Loaded {SUBJECT_ID} / {TARGET} / {SESSION}, x shape = {tuple(x.shape)}")

    g_0, C_base, u = encode_and_control(model, x)

    # --- Sanity check: at zero perturbation this must match the model's own
    # forward() reconstruction -- confirms manual_rollout_K's recurrence is
    # correctly re-implementing forward() before we trust anything built on it ---
    with torch.no_grad():
        real_out = model(x, item["lifus_condition"],
                          kl_g0_weight=1.0, kl_u_weight=1.0, apply_free_bits=False)
    zero_delta = torch.zeros(M, M, dtype=torch.complex64)
    schedule_all_false = [False] * T
    x_hat_check = manual_rollout_K(Lambda, g_0, C_base, u, W_bar_x, P_inv,
                                    zero_delta, schedule_all_false)
    max_diff = (x_hat_check - real_out["x_recon"]).abs().max().item()
    print(f"Sanity check -- max |manual_rollout_K(zero delta) - forward().x_recon| "
          f"= {max_diff:.2e}")
    if max_diff > 1e-3:
        print("  WARNING: manual rollout does not match forward() closely -- "
              "check the recurrence before trusting the perturbation results below.")

    # --- Decode K to ROI space, select source ROI, build the exact single-edge delta ---
    B = g_decode_matrix(W_bar_x, P_inv)
    K_roi_base, B_pinv = decode_K_to_roi(K_g, B)

    source_roi_idx = select_source_roi(
        K_roi_base, TARGET_ROI_IDX, override_name=SOURCE_ROI_NAME_OVERRIDE
    )
    print(f"\nPerturbing edge: {TARGET_ROIS[source_roi_idx]} -> {TARGET_ROI_NAME}, "
          f"requested magnitude = {EDGE_MAGNITUDE}")

    delta_K_g, delta_K_roi = build_delta_K_g(
        B, B_pinv, TARGET_ROI_IDX, source_roi_idx, EDGE_MAGNITUDE
    )

    # --- Diagnostic: does the raw decoded delta land EXACTLY at the target
    # cell and nowhere else, before any stability scaling or dynamics? ---
    K_roi_delta_check, _ = decode_K_to_roi(delta_K_g, B)
    off_target_leak = K_roi_delta_check.clone()
    off_target_leak[TARGET_ROI_IDX, source_roi_idx] = 0.0
    print(f"Static decode check -- max |leakage| into any OTHER K_roi cell "
          f"(should be ~0): {off_target_leak.abs().max().item():.2e}; "
          f"decoded value at target cell = "
          f"{K_roi_delta_check[TARGET_ROI_IDX, source_roi_idx].item():+.4f} "
          f"(requested {EDGE_MAGNITUDE:+.4f})")

    # --- Stability clamp: auto-scales the perturbation (not the underlying
    # K or Lambda) down if needed, before we ever run an unstable rollout ---
    scale, K_bar_delta = clamp_for_stability(Lambda, P, P_inv, delta_K_g)
    applied_magnitude = scale * EDGE_MAGNITUDE
    print(f"\nApplied edge magnitude after stability clamp: {applied_magnitude:+.4f} "
          f"({scale * 100:.1f}% of requested)")

    # --- Rollout: baseline (zero delta) vs. perturbed (clamped delta from onset) ---
    schedule = build_step_schedule()
    x_hat_base = manual_rollout_K(Lambda, g_0, C_base, u, W_bar_x, P_inv,
                                   zero_delta, schedule_all_false)
    x_hat_pert = manual_rollout_K(Lambda, g_0, C_base, u, W_bar_x, P_inv,
                                   K_bar_delta, schedule)

    effect = (x_hat_pert - x_hat_base).abs()
    print(f"\nEffect size on reconstructed BOLD: "
          f"max |Δx_hat| = {effect.max().item():.4f}, "
          f"mean |Δx_hat| = {effect.mean().item():.4f} "
          f"(raw BOLD is z-scored, so these are in std-dev units)")

    delta_np = (x_hat_pert - x_hat_base).numpy()
    print_impact_ranking(delta_np, source_roi_idx)

    # --- Plots (same style as the C script) ---
    out_path = OUT_DIR / f"K_edge_perturbation_{SUBJECT_ID}_{TARGET}_{SESSION}.png"
    plot_grid(x_raw=x.numpy(), x_hat_base=x_hat_base.numpy(),
              x_hat_pert=x_hat_pert.numpy(), source_roi_idx=source_roi_idx,
              out_path=out_path)

    delta_out_path = OUT_DIR / f"K_edge_perturbation_delta_{SUBJECT_ID}_{TARGET}_{SESSION}.png"
    plot_delta_grid(delta_np, source_roi_idx, delta_out_path)

    single_out_path = (OUT_DIR /
        f"K_edge_perturbation_single_{TARGET_ROI_NAME}_{SUBJECT_ID}_{TARGET}_{SESSION}.png")
    plot_single_roi_overlay(x_raw=x.numpy(), x_hat_base=x_hat_base.numpy(),
                             x_hat_pert=x_hat_pert.numpy(), roi_idx=TARGET_ROI_IDX,
                             source_roi_idx=source_roi_idx, out_path=single_out_path)


if __name__ == "__main__":
    main()