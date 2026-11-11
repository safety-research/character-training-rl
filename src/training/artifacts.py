"""Canonical on-disk artifact filenames written by every Tinker trainer.

Defined in exactly one place so the SFT, SDF, and RL trainers — and any
downstream loader that reads a run directory — agree on the layout. Renaming a
file here changes it everywhere at once; a literal copied into one trainer would
otherwise drift and leave a loader silently unable to find an artifact.
"""

from __future__ import annotations

# Windowed metric log (one record per log_every window).
TRAIN_LOG = "train_log.jsonl"
# Per-step raw metric log (one record per optimizer step).
TRAIN_STEPS_LOG = "train_steps.jsonl"
# Checkpoint manifest (sampler + state save paths, one record per save).
CHECKPOINTS_LOG = "checkpoints.jsonl"
# Effective run config, including resolved git commit and data digests.
CONFIG_JSON = "config.json"
# Append-only record of resume attempts (for auditing interrupted runs).
RESUME_ATTEMPTS_LOG = "resume_attempts.jsonl"
# Final-state pointer used to validate and drive a resume.
RESUME_META = "resume_meta.json"
# Weights & Biases run metadata (id/project/url) for the run, when W&B is on.
WANDB_JSON = "wandb.json"
