"""
================================================================================
BRICK Training Curve Plotter (single run directory)
================================================================================

Given ONE run directory (containing loss_history.csv plus one or both of
best_model_cls.pt / best_model_recon.pt), plots the 5 loss curves (Total,
Reconstruction, KL g0, KL u, Classification) in a 2x3 grid and saves the
figure.

Both checkpoint epochs are marked as vertical lines (cls and recon, in
different colors) so the best-recon and best-cls points are both visible on
every panel. A missing checkpoint is skipped with a note rather than an error.

OUTPUT PATH: preserves the subpath BETWEEN results/training and the run
directory, nested under results/figures/model_plots/. E.g.

    input:  results/training/ablation_2_batch_size_1/task_state_1/
    output: results/figures/model_plots/ablation_2_batch_size_1/task_state_1/training_curves.png

This preserves the full intermediate folder structure (not just the leaf
name) so runs with the same leaf name under different parents don't collide.

Usage:
    python analysis/create_model_plots.py <run_directory>
    python analysis/create_model_plots.py results/training/ablation_x/run_1
    python analysis/create_model_plots.py results/training/ablation_x/run_1 --y-max 5
    python analysis/create_model_plots.py results/training/ablation_x/run_1 \
        --cls-name best_model_cls.pt --recon-name best_model_recon.pt
"""

import sys
import csv
import argparse
from pathlib import Path

import torch
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

TRAINING_ROOT = ROOT_DIR / "results" / "training"
FIGURES_ROOT = ROOT_DIR / "results" / "figures" / "model_plots"

DEFAULT_CLS_NAME = "best_model_cls.pt"
DEFAULT_RECON_NAME = "best_model_recon.pt"
DEFAULT_Y_MAX = 10.0

# Colors for the two checkpoint marker lines.
CLS_COLOR = "red"
RECON_COLOR = "green"


# ================================================================================
# LOAD
# ================================================================================
def load_csv(csv_path: Path) -> dict:
    """Load loss history CSV into lists."""
    data = {k: [] for k in [
        "epochs",
        "train_total", "val_total",
        "train_recon", "val_recon",
        "train_kl_g0", "val_kl_g0",
        "train_kl_u",  "val_kl_u",
        "train_cls",   "val_cls",
    ]}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            data["epochs"].append(int(row["epoch"]))
            data["train_total"].append(float(row["train_loss_total"]))
            data["val_total"].append(float(row["val_loss_total"]))
            data["train_recon"].append(float(row["train_loss_recon"]))
            data["val_recon"].append(float(row["val_loss_recon"]))
            data["train_kl_g0"].append(float(row["train_loss_kl_g0"]))
            data["val_kl_g0"].append(float(row["val_loss_kl_g0"]))
            data["train_kl_u"].append(float(row["train_loss_kl_u"]))
            data["val_kl_u"].append(float(row["val_loss_kl_u"]))
            data["train_cls"].append(float(row["train_loss_cls"]))
            data["val_cls"].append(float(row["val_loss_cls"]))
    return data


def load_checkpoint_epoch(run_dir: Path, ckpt_name: str) -> int | None:
    """Return the stored 'epoch' from a checkpoint, or None if it's absent."""
    ckpt_path = run_dir / ckpt_name
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu")
        epoch = ckpt.get("epoch")
        print(f"  {ckpt_name}: best epoch = {epoch}")
        return epoch
    print(f"  {ckpt_name}: not found (skipping its marker line)")
    return None


# ================================================================================
# OUTPUT PATH
# ================================================================================
def derive_output_path(run_dir: Path) -> Path:
    """
    Preserve the subpath between results/training and the run directory,
    nested under results/figures/model_plots/.
        results/training/A/B/run  ->  results/figures/model_plots/A/B/run/training_curves.png
    Falls back to just the leaf name if run_dir is not under results/training.
    """
    run_dir = run_dir.resolve()
    try:
        rel = run_dir.relative_to(TRAINING_ROOT.resolve())
    except ValueError:
        print(f"  WARNING: {run_dir} is not under {TRAINING_ROOT} -- "
              f"using leaf folder name only for output path.")
        rel = Path(run_dir.name)
    out_dir = FIGURES_ROOT / rel
    return out_dir / "training_curves.png"


# ================================================================================
# PLOT
# ================================================================================
def plot_curves(data: dict, cls_epoch: int | None, recon_epoch: int | None,
                 run_name: str, out_path: Path, y_max: float | None = DEFAULT_Y_MAX):
    epochs = data["epochs"]

    def add_marker_lines(ax):
        # Draw both checkpoint markers; tolerate either being absent.
        if recon_epoch is not None:
            ax.axvline(x=recon_epoch, color=RECON_COLOR, linestyle=":", linewidth=1.5,
                       label=f"best recon (ep {recon_epoch})")
        if cls_epoch is not None:
            ax.axvline(x=cls_epoch, color=CLS_COLOR, linestyle=":", linewidth=1.5,
                       label=f"best cls (ep {cls_epoch})")

    def plot(ax, title, train, val=None, apply_ymax=False):
        ax.plot(epochs, train, label="train", linewidth=1.5)
        if val is not None:
            ax.plot(epochs, val, label="val", linewidth=1.5, linestyle="--")
        add_marker_lines(ax)
        if apply_ymax and y_max is not None:
            ax.set_ylim(0, y_max)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"BRICK Training Loss Curves\n{run_name}", fontsize=13)

    plot(axes[0, 0], "Total Loss",          data["train_total"], data["val_total"], apply_ymax=True)
    plot(axes[0, 1], "Reconstruction Loss", data["train_recon"], data["val_recon"], apply_ymax=True)
    plot(axes[0, 2], "KL g0",               data["train_kl_g0"], data["val_kl_g0"])
    plot(axes[1, 0], "KL u",                data["train_kl_u"],  data["val_kl_u"])
    plot(axes[1, 1], "Classification Loss", data["train_cls"],   data["val_cls"])
    axes[1, 2].axis("off")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")
    plt.close(fig)


# ================================================================================
# MAIN
# ================================================================================
def main():
    parser = argparse.ArgumentParser(description="Plot BRICK training curves for one run directory")
    parser.add_argument("run_dir", type=str,
                        help="Path to a single run directory (containing loss_history.csv)")
    parser.add_argument("--cls-name", type=str, default=DEFAULT_CLS_NAME,
                        help=f"Classification checkpoint filename (default: {DEFAULT_CLS_NAME})")
    parser.add_argument("--recon-name", type=str, default=DEFAULT_RECON_NAME,
                        help=f"Reconstruction checkpoint filename (default: {DEFAULT_RECON_NAME})")
    parser.add_argument("--y-max", type=float, default=DEFAULT_Y_MAX,
                        help=f"Y-axis cap for Total/Recon panels (default: {DEFAULT_Y_MAX})")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {run_dir}")

    csv_path = run_dir / "loss_history.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No loss_history.csv in {run_dir}")

    print(f"Plotting: {run_dir}")
    data = load_csv(csv_path)
    cls_epoch = load_checkpoint_epoch(run_dir, args.cls_name)
    recon_epoch = load_checkpoint_epoch(run_dir, args.recon_name)

    out_path = derive_output_path(run_dir)
    plot_curves(data, cls_epoch, recon_epoch, run_dir.name, out_path, y_max=args.y_max)


if __name__ == "__main__":
    main()