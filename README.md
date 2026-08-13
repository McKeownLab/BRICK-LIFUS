# LIFUS-BRICK

BRICK is a Koopman variational autoencoder for modeling resting-state fMRI dynamics. It decomposes brain dynamics into a shared, global linear operator (K) capturing baseline temporal dynamics, and a subject- and session-specific control signal (C) that carries any session-specific effect. This repository applies BRICK to a Parkinson's disease cohort (N=19) undergoing low-intensity focused ultrasound (LIFUS) neuromodulation targeting VIM and ZI. Model architecture and derivation follow Zhou et al. 2025, Section III.


## Repo structure
    config.py              — global hyperparameters
    models/                — brick.py, encoder.py, control.py, koopman_utils.py
    training/               — dataset.py, train.py, sweep.py, ablation_study.py, loso_train.py, train_task_states.py
    analysis/               — pre/post stats, LOSO analysis, perturbation analysis
    preprocessing/          — data loading, ROI definitions
    results/                — training runs, figures


## Model overview

**`config.py`** — All hyperparameters and shared constants (model dimensions, training defaults, paths) live here, so that runs stay consistent across training, sweeps, and analysis without duplicating values.

**`models/`** — The core BRICK implementation: the Encoder (posterior over the initial latent state g_0), the ControlModule (generates the diagonal control matrix C, control inputs u, and the task-state classifier), and the Koopman utilities (computing K = P @ diag(Lambda) @ P_inv and the eigenspace recurrence). This is the model architecture itself, independent of any particular dataset.

**`preprocessing/`** — Preprocessing specific to our data: loading raw BOLD timeseries, defining the target ROI set, and any subject-level formatting needed before data reaches `training/`.

**`training/`** — Everything needed to train BRICK. `train.py` is the main entry point (also defines the `BRICKDataset` class used across training and analysis); alongside it are variant training scripts for different setups (e.g. batched vs. non-batched runs), plus sweep and ablation-study scripts for running and comparing multiple configurations.

**`analysis/`** — All analysis performed on our trained models and data: pre/post statistical testing, LOSO evaluation, perturbation/stability analysis, seed and batch-size robustness checks, and the descriptive K comparisons. NOTE: many directory names have changed since creation and may need to be edited.

**`results/`** — Output of the above: trained model checkpoints, loss histories, and generated figures, organized by run/sweep/ablation name. 


## Citation

This repository implements and extends the BRICK model from:

> Z. Zhou, T. Dan, and G. Wu, "Understanding Brain Functional Dynamics
> Through Neural Koopman Operator with Control Mechanism," *IEEE
> Transactions on Medical Imaging*, vol. 44, no. 11, pp. 4627–4638,
> Nov. 2025. doi: 10.1109/TMI.2025.3580611
