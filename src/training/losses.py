"""The training loss math, in one place.

This is the "last calculation" each trainer performs per step: SFT/SDF weighted
cross-entropy NLL, and the DPO / KTO preference losses for RL. Kept separate
from the trainers so the formulas are readable and independently tested. Torch is
imported lazily so the data-only code paths do not require it.
"""

from __future__ import annotations

import math
from typing import Any

from src.training.arrays import values_to_list
from src.training.tinker_common import tensor_values

# KTO clamps the batch KL baseline at zero so a (noisy) negative KL estimate cannot
# turn into a reward bonus. Zero is the KTO definition's floor, not a tunable.
KTO_KL_FLOOR = 0.0


def mean_weighted_nll(fwd_bwd_result, batch: list) -> float:
    """Weighted mean negative log-likelihood from Tinker's per-token logprobs.

    Averages -logprob over target tokens, weighted by each token's loss weight, so
    masked (weight 0) tokens do not count. Raises if the batch has no trained tokens.
    """
    total_weighted_logprob = 0.0
    total_weight = 0.0
    for output, datum in zip(fwd_bwd_result.loss_fn_outputs, batch, strict=True):
        logprobs = tensor_values(output["logprobs"])
        weights = tensor_values(datum.loss_fn_inputs["weights"])
        if len(logprobs) != len(weights):
            raise ValueError(f"forward_backward returned {len(logprobs)} logprobs for {len(weights)} weights")
        total_weighted_logprob += sum(float(logprob) * float(weight) for logprob, weight in zip(logprobs, weights))
        total_weight += sum(float(weight) for weight in weights)
    if total_weight <= 0:
        raise ValueError("batch has no trained tokens; refusing to report zero loss")
    nll = -total_weighted_logprob / total_weight
    if not math.isfinite(nll):
        raise ValueError(f"weighted NLL is not finite ({nll}); the run has diverged or returned bad logprobs")
    return nll


def _tensor(values):
    """Coerce values (TensorData / torch / list) to a detached float32 torch tensor."""
    import torch

    if hasattr(values, "detach"):
        return values.detach() if not values.requires_grad else values
    return torch.tensor(values_to_list(values), dtype=torch.float32)


def compute_dpo_loss(
    chosen_logprobs: list,
    rejected_logprobs: list,
    chosen_ref_logprobs: list,
    rejected_ref_logprobs: list,
    chosen_weights: list,
    rejected_weights: list,
    dpo_beta: float,
) -> tuple[Any, dict[str, float]]:
    """DPO loss over prompt-masked log probabilities (delegates the core to the cookbook)."""
    import torch

    chosen_policy = []
    rejected_policy = []
    chosen_reference = []
    rejected_reference = []
    for i in range(len(chosen_logprobs)):
        chosen_policy.append(torch.dot(_tensor(chosen_logprobs[i]), chosen_weights[i]))
        chosen_reference.append(torch.dot(chosen_ref_logprobs[i], chosen_weights[i]))
        rejected_policy.append(torch.dot(_tensor(rejected_logprobs[i]), rejected_weights[i]))
        rejected_reference.append(torch.dot(rejected_ref_logprobs[i], rejected_weights[i]))

    from tinker_cookbook.preference.train_dpo import compute_dpo_loss as cookbook_compute_dpo_loss

    loss, metrics = cookbook_compute_dpo_loss(
        chosen_logprobs=chosen_policy,
        rejected_logprobs=rejected_policy,
        chosen_ref_logprobs=chosen_reference,
        rejected_ref_logprobs=rejected_reference,
        dpo_beta=dpo_beta,
    )
    metrics["loss"] = metrics["dpo_loss"]
    return loss, metrics


def compute_kto_loss(
    logprobs: list,
    ref_logprobs: list,
    weights: list,
    desirable_flags: list[bool],
    kto_beta: float,
    desirable_weight: float,
    undesirable_weight: float,
    kl_logratios: list | None = None,
) -> tuple[Any, dict[str, float]]:
    """KTO loss (Ethayarajh et al. 2024, eq. 7), matching the reference TRL implementation.

    Per example the implicit reward is r = kto_beta * (log pi(y|x) - log pi_ref(y|x)) summed over
    the trained (weight > 0) tokens. The reference point z0 is the batch KL estimate: the mean of
    `kl_logratios`, which are UNSCALED policy-minus-reference log-ratios of mismatched (x_i, y_j)
    sequences, detached and clamped at zero. Desirable examples are pushed above z0 and
    undesirable ones below it:

        loss_i = w(y_i) * (1 - sigmoid(kto_beta * (logratio_i - z0)))      if desirable
        loss_i = w(y_i) * (1 - sigmoid(kto_beta * (z0 - logratio_i)))      if undesirable

    and the batch loss is the mean over all examples (not a per-class mean), so w(y) trades off
    against the class counts exactly as the TRL guidance assumes: choose the weights so that
    desirable_weight * n_desirable : undesirable_weight * n_undesirable is between 1:1 and 4:3.
    A batch may contain a single class; the dataset must contain both.
    """
    import torch

    if kl_logratios is None:
        raise ValueError("KTO requires explicit mismatched-pair KL log-ratio samples")
    if not (len(logprobs) == len(ref_logprobs) == len(weights) == len(desirable_flags)):
        raise ValueError("KTO inputs must be aligned per example")
    if not logprobs:
        raise ValueError("KTO batch is empty")

    log_ratios = []
    for i in range(len(logprobs)):
        model_lp = torch.dot(_tensor(logprobs[i]), weights[i])
        ref_lp = torch.dot(ref_logprobs[i], weights[i])
        log_ratios.append(model_lp - ref_lp)
    log_ratios_t = torch.stack(log_ratios)
    kl = torch.stack([_tensor(value).reshape(()) for value in kl_logratios]).detach().mean().clamp(min=KTO_KL_FLOOR)

    desirable_mask = torch.tensor(desirable_flags, dtype=torch.bool, device=log_ratios_t.device)
    rewards = kto_beta * log_ratios_t
    kl_reward = kto_beta * kl
    per_example = torch.where(
        desirable_mask,
        desirable_weight * (1.0 - torch.sigmoid(rewards - kl_reward)),
        undesirable_weight * (1.0 - torch.sigmoid(kl_reward - rewards)),
    )
    loss = per_example.mean()

    n_desirable = int(desirable_mask.sum().item())
    n_undesirable = len(desirable_flags) - n_desirable
    detached_rewards = rewards.detach()
    desirable_rewards = detached_rewards[desirable_mask]
    undesirable_rewards = detached_rewards[~desirable_mask]
    desirable_accuracy = float((desirable_rewards > 0).float().mean().item()) if n_desirable else None
    undesirable_accuracy = float((undesirable_rewards < 0).float().mean().item()) if n_undesirable else None
    reward_desirable = float(desirable_rewards.mean().item()) if n_desirable else None
    reward_undesirable = float(undesirable_rewards.mean().item()) if n_undesirable else None
    reward_margin = (
        reward_desirable - reward_undesirable
        if reward_desirable is not None and reward_undesirable is not None
        else None
    )

    kto_loss = float(loss.detach().item())
    return loss, {
        "kto_loss": kto_loss,
        "loss": kto_loss,
        "kl": float(kl.item()),
        "log_ratio_mean": float(log_ratios_t.mean().detach().item()),
        "reward_desirable": reward_desirable,
        "reward_undesirable": reward_undesirable,
        "reward_margin": reward_margin,
        "desirable_accuracy": desirable_accuracy,
        "undesirable_accuracy": undesirable_accuracy,
        "n_desirable": n_desirable,
        "n_undesirable": n_undesirable,
    }


def require_step_loss(metrics: dict[str, Any], key: str) -> float:
    """Return metrics[key] as a finite float, raising if missing, non-numeric, or non-finite."""
    if key not in metrics:
        raise ValueError(f"required training metric {key!r} missing from step metrics: {sorted(metrics)}")
    value = metrics[key]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"required training metric {key!r} must be numeric, got {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"required training metric {key!r} must be finite, got {value!r}")
    return value
