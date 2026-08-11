"""
================================================================================
Pre vs. Post Sonication: C vs. u_t Tradeoff, Pooled Across Patients & ROIs
================================================================================

Population-level version of compare_C_and_u_tradeoff.py. Two changes from
that single-subject, per-mode version:

    1. POOLED ACROSS PATIENTS: loops over every completed LOSO fold, excluding
       EXCLUDED_SUBJECTS

    2. DECODED TO ROI SPACE, NOT LEFT IN MODE SPACE: C and u_t are decoded
       from 96 modes to 24 ROIs via A = Re(W_bar_x) BEFORE computing
       magnitude/RMS (decode-then-magnitude, not the reverse -- A is only a
       meaningful linear operator on the model's actual signed state, not
       on a rectified/magnitude quantity). W_bar_x is NOT shared across
       folds -- each LOSO fold is a separately-trained model, so A is
       recomputed per-subject from that subject's own checkpoint.

    Each point in the resulting scatter is one (patient, ROI) pair --
    N_subjects x 24 ROIs.

WHY RAW DIFFERENCE, NOT %-CHANGE: the single-subject version used %-change
and produced a very unstable-looking plot -- one mode had |C| near zero
pre-sonication, so a small post value produced a ~1600% change. Pooling
across patients would make this worse (more chances to divide by
near-zero). Switched to raw difference (post - pre) for the tradeoff
scatter -- Pearson correlation is scale-invariant, so it still answers the
tradeoff-vs-co-movement question without the blow-up.

FRAMING NOTE (same as compare_C_t.py / compare_C_and_u_tradeoff.py): |C|
is unsigned magnitude. The term RMS uses the SIGNED product C[m]*u_t[t,m]
(sign matters for the actual driving signal), decoded to ROI space before
taking RMS over time.

Usage:
    python analysis/compare_C_u_tradeoff_population.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import stats as sstats

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from preprocessing.load_preprocessed_data import TARGET_ROIS
from analysis.analysis_helper_functions import load_model, compute_K
from training.dataset import BRICKDataset
from training.train import DATA_DIR

# ================================================================================
# CONFIG
# ================================================================================

# Change DIR
LOSO_DIR = ROOT_DIR / "results" / "training" / "loso_19_fold_beta_0.2"

# Toggle: True = analyze in raw 96-mode space (no ROI decode -- robustness
# check that sidesteps the A=Re(W_bar_x) decode assumption); False = decode
# to 24 ROIs first (the interpretable version).
USE_MODE_SPACE = True

_SPACE_TAG = "modes" if USE_MODE_SPACE else "rois"
_UNIT_LABEL = "mode" if USE_MODE_SPACE else "ROI"
_DECODE_LABEL = "raw mode-space" if USE_MODE_SPACE else "ROI-decoded"
OUT_DIR = ROOT_DIR / "results" / "figures" / f"C_u_tradeoff_population_{_SPACE_TAG}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXCLUDED_SUBJECTS = {"sub-fuspd09", "sub-fuspd15", "sub-fuspd19"}  # did not converge

CONDITION_TO_SESSION = {"mpre": "pre", "mpost": "post"}
TARGET_FIRST_LABEL = {"vim": "VIM_first", "zi": "ZI_first"}
TARGET_COLOR = {"vim": "#4C72B0", "zi": "#DD8452"}


# ================================================================================
# 1. DISCOVER FOLDS
# ================================================================================
def find_folds():
    """{subject_id: checkpoint_path} for every completed LOSO fold with a
    best_model_cls.pt, excluding EXCLUDED_SUBJECTS. Same convention as
    loso_analyze.py's find_folds()."""
    folds = {}
    for d in sorted(LOSO_DIR.glob("fold_*")):
        if not d.is_dir():
            continue
        subject_id = d.name[len("fold_"):]
        if subject_id in EXCLUDED_SUBJECTS:
            continue
        ckpt = d / "best_model_cls.pt"
        if ckpt.exists():
            folds[subject_id] = ckpt
    return folds


def determine_first_treatment(ds, subject_id):
    for i in range(len(ds)):
        item = ds[i]
        if item["subject_id"] == subject_id:
            group_str = item["group_str"]
            for target, label in TARGET_FIRST_LABEL.items():
                if group_str == label:
                    return target
            raise ValueError(f"Unrecognized group_str {group_str!r} for {subject_id}")
    raise ValueError(f"No items found for subject_id={subject_id!r}")


def load_pre_post_items(ds, subject_id, target):
    pre_item, post_item = None, None
    for i in range(len(ds)):
        item = ds[i]
        if item["subject_id"] == subject_id and item["target"] == target:
            session = CONDITION_TO_SESSION[item["condition_str"]]
            if session == "pre":
                pre_item = item
            elif session == "post":
                post_item = item
    if pre_item is None or post_item is None:
        raise ValueError(f"Missing pre and/or post item for {subject_id}/{target}")
    return pre_item, post_item


# ================================================================================
# 2. PER-SUBJECT COMPUTATION
# ================================================================================
def to_complex_tensor(x):
    """compute_K() numpy-ifies its outputs -- cast back to complex64 torch."""
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(torch.complex64)
    if torch.is_tensor(x):
        return x.to(torch.complex64)
    return torch.as_tensor(x, dtype=torch.complex64)


def roi_decode_matrix(W_bar_x):
    """A = Re(W_bar_x), shape (N_ROIS, M) real. See ASSUMPTION note in
    module docstring."""
    return W_bar_x.real


def compute_C_and_u(model, x):
    """Returns (C_diag, u): C_diag signed (M,) numpy, u (T, M) real numpy."""
    with torch.no_grad():
        C, u, s_hat, mu_u, logvar_u = model.control(x)
    c_diag = torch.diagonal(C)
    if c_diag.is_complex():
        c_diag = c_diag.real
    return c_diag.numpy(), u.numpy()


def rms_over_time(x_2d):
    """(T, N) -> (N,), RMS over axis 0."""
    return np.sqrt(np.mean(x_2d ** 2, axis=0))


def process_subject(subject_id, checkpoint_path, ds):
    model = load_model(checkpoint_path)
    K, Lambda, W_bar_x = compute_K(model)
    W_bar_x = to_complex_tensor(W_bar_x)

    target = determine_first_treatment(ds, subject_id)
    pre_item, post_item = load_pre_post_items(ds, subject_id, target)

    C_pre_diag, u_pre = compute_C_and_u(model, pre_item["x"])     # (M,), (T,M)
    C_post_diag, u_post = compute_C_and_u(model, post_item["x"])

    if USE_MODE_SPACE:
        # Raw 96-mode space -- no decode at all. Sidesteps the A=Re(W_bar_x)
        # assumption entirely, as a robustness check on the ROI-space result.
        # "units" here are the 96 modes rather than 24 ROIs.
        unit_names = [f"mode_{m}" for m in range(len(C_pre_diag))]
        C_mag_pre = np.abs(C_pre_diag)                 # (M,)
        C_mag_post = np.abs(C_post_diag)
        u_rms_pre = rms_over_time(u_pre)               # (M,)
        u_rms_post = rms_over_time(u_post)
        term_pre = u_pre * C_pre_diag[None, :]         # (T, M)
        term_post = u_post * C_post_diag[None, :]
        term_rms_pre = rms_over_time(term_pre)         # (M,)
        term_rms_post = rms_over_time(term_post)
    else:
        # ROI space: decode-then-magnitude (decode signed quantities via A).
        A = roi_decode_matrix(W_bar_x).numpy()         # (N_ROIS, M) real
        unit_names = list(TARGET_ROIS)
        C_mag_pre = np.abs(A @ C_pre_diag)             # (N_ROIS,)
        C_mag_post = np.abs(A @ C_post_diag)
        u_rms_pre = rms_over_time(u_pre @ A.T)         # (N_ROIS,)
        u_rms_post = rms_over_time(u_post @ A.T)
        # Whole term: exact elementwise product in mode space (C diagonal),
        # THEN decode the resulting time series (decode is linear).
        term_pre_modes = u_pre * C_pre_diag[None, :]
        term_post_modes = u_post * C_post_diag[None, :]
        term_rms_pre = rms_over_time(term_pre_modes @ A.T)   # (N_ROIS,)
        term_rms_post = rms_over_time(term_post_modes @ A.T)

    rows = []
    for i, unit_name in enumerate(unit_names):
        rows.append({
            "subject_id": subject_id, "target": target, "roi": unit_name,
            "C_mag_pre": C_mag_pre[i], "C_mag_post": C_mag_post[i],
            "u_rms_pre": u_rms_pre[i], "u_rms_post": u_rms_post[i],
            "term_rms_pre": term_rms_pre[i], "term_rms_post": term_rms_post[i],
        })
    return rows


# ================================================================================
# 3. PLOTS
# ================================================================================
def plot_C_vs_u_scatter(rows, out_path):
    """
    C on the x-axis, u_t on the y-axis, raw values (not deltas) -- pre and
    post as two colors, pooled across all (patient, ROI) points. Shows
    directly whether the pre cloud and post cloud sit in different parts
    of the C-u_t plane (shifted right = C bigger, shifted up = u_t bigger,
    shifted diagonally = both together, no shift = neither).
    """
    C_pre = np.array([r["C_mag_pre"] for r in rows])
    C_post = np.array([r["C_mag_post"] for r in rows])
    u_pre = np.array([r["u_rms_pre"] for r in rows])
    u_post = np.array([r["u_rms_post"] for r in rows])

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(C_pre, u_pre, color="#4C72B0", alpha=0.5, label="pre-sonication")
    ax.scatter(C_post, u_post, color="#DD8452", alpha=0.5, label="post-sonication")

    ax.set_xlabel(r"$|C|$ (" + _DECODE_LABEL + ")")
    ax.set_ylabel(r"RMS($u_t$) (" + _DECODE_LABEL + ")")
    ax.set_title(f"C vs. u_t, pooled across patients x {_UNIT_LABEL}s (N={len(rows)} points each)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


def plot_term_paired_scatter(rows, out_path):
    """Paired scatter (pre on x, post on y) of ROI-decoded RMS(C*u_t),
    pooled across patients x {_UNIT_LABEL}s, with a y=x reference line."""
    term_pre = np.array([r["term_rms_pre"] for r in rows])
    term_post = np.array([r["term_rms_post"] for r in rows])
    targets = [r["target"] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 8))
    for target, color in TARGET_COLOR.items():
        mask = np.array([t == target for t in targets])
        ax.scatter(term_pre[mask], term_post[mask], color=color, alpha=0.5,
                   label=f"{target.upper()} (1st tx)")

    lims = [0, max(term_pre.max(), term_post.max()) * 1.05]
    ax.plot(lims, lims, color="black", lw=1.0, linestyle="--", label="y = x")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_aspect("equal")

    ax.set_xlabel(r"RMS($C \cdot u_t$), pre (" + _DECODE_LABEL + ")")
    ax.set_ylabel(r"RMS($C \cdot u_t$), post (" + _DECODE_LABEL + ")")
    ax.set_title(f"Whole-term magnitude, pooled across patients x {_UNIT_LABEL}s (N={len(rows)})")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)
    return term_pre, term_post


def plot_compensation_analysis(rows, out_path):
    """
    Direct test of "does bigger C mean u_t compensates (smaller), keeping
    C*u_t about the same -- or does the term itself get bigger with C?"
    Pools pre AND post as separate observations of the general C-u_t
    relationship.

    Left panel: u_t vs C. Negative correlation = compensation (bigger C,
    smaller u_t). No/positive correlation = no compensation.

    Right panel: the term (C*u_t) vs C. A flat (near-zero slope) line means 
    the term stays about the same size regardless of C (compensation exactly 
    cancels C's growth). A slope close to mean(u_t) means the term scales 
    roughly linearly with C, i.e. u_t is NOT compensating.
    """
    C_all, u_all, term_all, session_all = [], [], [], []
    for r in rows:
        C_all.append(r["C_mag_pre"]);  u_all.append(r["u_rms_pre"]);  term_all.append(r["term_rms_pre"]);  session_all.append("pre")
        C_all.append(r["C_mag_post"]); u_all.append(r["u_rms_post"]); term_all.append(r["term_rms_post"]); session_all.append("post")
    C_all = np.array(C_all); u_all = np.array(u_all); term_all = np.array(term_all)
    is_pre = np.array([s == "pre" for s in session_all])

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # --- Left: u_t vs C ---
    ax = axes[0]
    ax.scatter(C_all[is_pre], u_all[is_pre], color="#4C72B0", alpha=0.4, s=20, label="pre")
    ax.scatter(C_all[~is_pre], u_all[~is_pre], color="#DD8452", alpha=0.4, s=20, label="post")
    slope_u, intercept_u, r_u, p_u, se_u = sstats.linregress(C_all, u_all)

    ax.set_xlabel(r"$|C|$ (" + _DECODE_LABEL + ")")
    ax.set_ylabel(r"RMS($u_t$) (" + _DECODE_LABEL + ")")
    ax.set_title(f"u_t vs C\nregression slope={slope_u:.4f}, r={r_u:.3f} (p={p_u:.3g})",
                 fontsize=10)
    ax.legend(fontsize=8)

    # --- Right: term (C*u_t) vs C -- the direct test ---
    ax = axes[1]
    ax.scatter(C_all[is_pre], term_all[is_pre], color="#4C72B0", alpha=0.4, s=20, label="pre")
    ax.scatter(C_all[~is_pre], term_all[~is_pre], color="#DD8452", alpha=0.4, s=20, label="post")
    slope_t, intercept_t, r_t, p_t, se_t = sstats.linregress(C_all, term_all)
    xs_line = np.linspace(C_all.min(), C_all.max(), 100)
    ax.plot(xs_line, slope_t * xs_line + intercept_t, color="black", lw=1.5)
    ax.set_xlabel(r"$|C|$ (" + _DECODE_LABEL + ")")
    ax.set_ylabel(r"RMS($C \cdot u_t$) (the term, " + _DECODE_LABEL + ")")
    ax.set_title(f"term (C*u_t) vs C\nslope={slope_t:.4f}, r={r_t:.3f} (p={p_t:.3g})\n"
                 f"mean(u_t)={u_all.mean():.4f}  (slope near this = no compensation)",
                 fontsize=10)
    ax.legend(fontsize=8)

    fig.suptitle(f"Does u_t compensate for C, or does the term scale with C? (N={len(C_all)} points, pre+post pooled)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)

    print(f"\nCompensation analysis (N={len(C_all)}, pre+post pooled as separate observations):")
    print(f"  u_t vs C:      slope={slope_u:.4f}  r={r_u:.3f}  p={p_u:.3g}")
    print(f"  term vs C:     slope={slope_t:.4f}  r={r_t:.3f}  p={p_t:.3g}   (mean(u_t)={u_all.mean():.4f})")
    if r_u < -0.1 and p_u < 0.05 and abs(r_t) < 0.1:
        print(f"  -> COMPENSATION: u_t decreases as C increases, term stays roughly flat.")
    elif r_t > 0.1 and p_t < 0.05:
        print(f"  -> NO COMPENSATION: term increases with C -- bigger C means more signal "
              f"getting through, not offset by smaller u_t.")
    else:
        print(f"  -> Ambiguous / weak effects either way -- check the plot directly.")



def plot_compensation_analysis_demeaned(rows, out_path):
    """
    Same question as plot_compensation_analysis(), but demeaned per subject
    first: for each subject, subtract that subject's own mean C / u_t /
    term (computed across all 48 of their observations -- 24 ROIs x
    pre+post) before pooling across subjects. This removes each patient's
    baseline scale (which the raw-value analysis showed varies a lot
    between LOSO folds -- e.g. sub-fuspd07's u_t std is ~3x sub-fuspd09's,
    despite identical z-scored input, since each fold is a separately
    trained model) and isolates the within-patient, across-region
    relationship -- exactly the axis a naive pooled correlation can mask.
    """
    by_subject = {}
    for r in rows:
        by_subject.setdefault(r["subject_id"], []).append(r)

    C_all, u_all, term_all, session_all = [], [], [], []
    for subject_id, sub_rows in by_subject.items():
        C_vals, u_vals, term_vals, sess_vals = [], [], [], []
        for r in sub_rows:
            C_vals += [r["C_mag_pre"], r["C_mag_post"]]
            u_vals += [r["u_rms_pre"], r["u_rms_post"]]
            term_vals += [r["term_rms_pre"], r["term_rms_post"]]
            sess_vals += ["pre", "post"]
        C_vals = np.array(C_vals); u_vals = np.array(u_vals); term_vals = np.array(term_vals)

        C_all.append(C_vals - C_vals.mean())
        u_all.append(u_vals - u_vals.mean())
        term_all.append(term_vals - term_vals.mean())
        session_all += sess_vals

    C_all = np.concatenate(C_all); u_all = np.concatenate(u_all); term_all = np.concatenate(term_all)
    is_pre = np.array([s == "pre" for s in session_all])

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    ax = axes[0]
    ax.scatter(C_all[is_pre], u_all[is_pre], color="#4C72B0", alpha=0.4, s=20, label="pre")
    ax.scatter(C_all[~is_pre], u_all[~is_pre], color="#DD8452", alpha=0.4, s=20, label="post")
    slope_u, intercept_u, r_u, p_u, se_u = sstats.linregress(C_all, u_all)
    xs = np.linspace(C_all.min(), C_all.max(), 100)
    ax.plot(xs, slope_u * xs + intercept_u, color="black", lw=1.5)
    ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel(r"$|C|$ minus subject's own mean")
    ax.set_ylabel(r"RMS($u_t$) minus subject's own mean")
    ax.set_title(f"u_t vs C (demeaned per subject)\nslope={slope_u:.4f}, r={r_u:.3f} (p={p_u:.3g})\n"
                 f"{'compensation (negative)' if (r_u < 0 and p_u < 0.05) else 'no evidence of compensation'}",
                 fontsize=10)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.scatter(C_all[is_pre], term_all[is_pre], color="#4C72B0", alpha=0.4, s=20, label="pre")
    ax.scatter(C_all[~is_pre], term_all[~is_pre], color="#DD8452", alpha=0.4, s=20, label="post")
    slope_t, intercept_t, r_t, p_t, se_t = sstats.linregress(C_all, term_all)
    ax.plot(xs, slope_t * xs + intercept_t, color="black", lw=1.5)
    ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel(r"$|C|$ minus subject's own mean")
    ax.set_ylabel(r"RMS($C \cdot u_t$) minus subject's own mean")
    ax.set_title(f"term (C*u_t) vs C (demeaned per subject)\nslope={slope_t:.4f}, r={r_t:.3f} (p={p_t:.3g})",
                 fontsize=10)
    ax.legend(fontsize=8)

    fig.suptitle(f"Same question, demeaned per subject first (N={len(C_all)} points, "
                 f"{len(by_subject)} subjects)", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)

    print(f"\nCompensation analysis, DEMEANED PER SUBJECT (N={len(C_all)}, {len(by_subject)} subjects):")
    print(f"  u_t vs C:      slope={slope_u:.4f}  r={r_u:.3f}  p={p_u:.3g}")
    print(f"  term vs C:     slope={slope_t:.4f}  r={r_t:.3f}  p={p_t:.3g}")
    if r_u < -0.1 and p_u < 0.05 and abs(r_t) < 0.1:
        print(f"  -> COMPENSATION (within-patient): u_t decreases as C increases across "
              f"a patient's own regions, term stays roughly flat.")
    elif r_t > 0.1 and p_t < 0.05:
        print(f"  -> NO COMPENSATION (within-patient): term increases with C across regions.")
    else:
        print(f"  -> Still ambiguous / weak even after removing between-patient scale "
              f"differences -- check the plot directly.")


def print_stats(rows, tradeoff_r, tradeoff_p, term_pre, term_post):
    n_subjects = len(set(r["subject_id"] for r in rows))
    print(f"\nPooled across {n_subjects} subjects x {_UNIT_LABEL}s = {len(rows)} points")

    print(f"\nTradeoff correlation (delta|C| vs delta RMS(u_t)): "
          f"r={tradeoff_r:.3f}, p={tradeoff_p:.3g}")
    if tradeoff_p < 0.05:
        direction = "TRADEOFF (negative correlation)" if tradeoff_r < 0 else "CO-MOVEMENT (positive correlation)"
        print(f"  -> significant: {direction}")
    else:
        print(f"  -> not significant: C and u_t changes appear largely independent")

    t_stat, t_p = sstats.ttest_rel(term_post, term_pre)
    w_stat, w_p = sstats.wilcoxon(term_post, term_pre)
    print(f"\nWhole term RMS(C*u_t), paired (post vs pre, pooled N={len(rows)}):")
    print(f"  Paired t-test: t={t_stat:.3f}, p={t_p:.4g}")
    print(f"  Wilcoxon signed-rank: stat={w_stat:.2f}, p={w_p:.4g}")
    print(f"  NOTE: points are not independent -- {_UNIT_LABEL}s within a subject "
          f"share that subject's overall scale, and subjects vary too. "
          f"Treat these pooled tests as descriptive/exploratory, not a "
          f"rigorous population-level test (that would need the subject-"
          f"level paired framework used elsewhere in this project).")


# ================================================================================
# MAIN
# ================================================================================
def main():
    folds = find_folds()
    print(f"Found {len(folds)} completed fold(s) (after excluding {sorted(EXCLUDED_SUBJECTS)}): "
          f"{sorted(folds.keys())}")

    ds = BRICKDataset(DATA_DIR)

    all_rows = []
    for subject_id, checkpoint_path in folds.items():
        print(f"Processing {subject_id}...")
        rows = process_subject(subject_id, checkpoint_path, ds)
        all_rows.extend(rows)

    plot_C_vs_u_scatter(
        all_rows, OUT_DIR / "C_vs_u_scatter_pooled.png"
    )
    term_pre, term_post = plot_term_paired_scatter(
        all_rows, OUT_DIR / "term_paired_scatter_pooled.png"
    )
    plot_compensation_analysis(
        all_rows, OUT_DIR / "compensation_analysis.png"
    )
    plot_compensation_analysis_demeaned(
        all_rows, OUT_DIR / "compensation_analysis_demeaned.png"
    )

    # Tradeoff correlation is still a useful summary stat even though the
    # plot itself now shows raw values, not deltas -- computed separately here.
    delta_C = np.array([row["C_mag_post"] - row["C_mag_pre"] for row in all_rows])
    delta_u = np.array([row["u_rms_post"] - row["u_rms_pre"] for row in all_rows])
    r, p = sstats.pearsonr(delta_C, delta_u)

    print_stats(all_rows, r, p, term_pre, term_post)


if __name__ == "__main__":
    main()