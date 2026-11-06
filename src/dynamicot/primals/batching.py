from __future__ import annotations

import torch


def sample_training_batch(
    source_data: torch.Tensor,
    target_data: torch.Tensor,
    *,
    step: int,
    batch_size: int,
    batching_type: str,
    state: dict[str, torch.Tensor] | None = None,
):
    """'random' shuffles source and target independently each epoch; 'paired' keeps the
    given source/target pairing (used for paired translation tasks)."""
    if batching_type not in {"random", "paired"}:
        raise ValueError("batching_type must be one of: random, paired")

    state = {} if state is None else dict(state)
    n_samples = source_data.shape[0]
    batches_per_epoch = n_samples // batch_size
    batch_in_epoch = step % batches_per_epoch
    device = source_data.device

    if batching_type == "random":
        if batch_in_epoch == 0 or "source_shuffled" not in state:
            state["source_shuffled"] = source_data[torch.randperm(n_samples, device=device)]
            state["target_shuffled"] = target_data[torch.randperm(n_samples, device=device)]
        start = batch_in_epoch * batch_size
        end = start + batch_size
        return (
            state["source_shuffled"][start:end].detach().clone(),
            state["target_shuffled"][start:end].detach().clone(),
            state,
        )

    if batch_in_epoch == 0 or "source_shuffled" not in state:
        perm = torch.randperm(n_samples, device=device)
        state["source_shuffled"] = source_data[perm]
        state["target_shuffled"] = target_data[perm]
    start = batch_in_epoch * batch_size
    end = start + batch_size
    return (
        state["source_shuffled"][start:end].detach().clone(),
        state["target_shuffled"][start:end].detach().clone(),
        state,
    )
