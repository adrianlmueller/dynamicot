from __future__ import annotations

import torch

from dynamicot.duals.objectives import grad_cost_x, grad_cost_y


@torch.no_grad()
def ot_permutation(x_batch: torch.Tensor, y_batch: torch.Tensor, p: float = 2.0):
    from scipy.optimize import linear_sum_assignment

    batch_size = x_batch.shape[0]
    device = y_batch.device
    x_work = x_batch.detach().to(dtype=torch.float32)
    y_work = y_batch.detach().to(dtype=torch.float32)
    cost = torch.cdist(x_work, y_work, p=p).square()
    row_ind, col_ind = linear_sum_assignment(cost.cpu().numpy())

    perm = torch.empty(batch_size, dtype=torch.long, device=device)
    row_ind = torch.as_tensor(row_ind, dtype=torch.long, device=device)
    col_ind = torch.as_tensor(col_ind, dtype=torch.long, device=device)
    perm[row_ind] = col_ind
    return perm


def apply_initialization_coupling(
    x_batch: torch.Tensor,
    y_batch: torch.Tensor,
    *,
    initialization_type: str,
):
    """Couple the minibatch used for particle initialization: 'linear' keeps the given
    (random) pairing, 'ot' re-pairs the batch with an exact minibatch OT assignment."""
    initialization_type = initialization_type.lower()
    if initialization_type == "linear":
        return x_batch, y_batch
    if initialization_type == "ot":
        return x_batch, y_batch[ot_permutation(x_batch, y_batch, p=2.0)].detach().clone()
    raise ValueError("initialization_type must be one of: linear, ot")


def build_time_features(
    *,
    horizon: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
):
    h_minus = torch.arange(horizon + 1, device=device, dtype=dtype) / (horizon + 1)
    h_plus = torch.arange(1, horizon + 2, device=device, dtype=dtype) / (horizon + 1)
    h_feat = torch.arange(1, horizon + 1, device=device, dtype=dtype) / (horizon + 1)
    return (
        h_minus.repeat_interleave(batch_size).unsqueeze(1),
        h_plus.repeat_interleave(batch_size).unsqueeze(1),
        h_feat.repeat_interleave(batch_size).unsqueeze(1),
    )


def initialize_particle_pairs(
    x_batch: torch.Tensor,
    y_batch: torch.Tensor,
    *,
    horizon: int,
):
    h_grid = torch.arange(horizon + 1, device=x_batch.device, dtype=x_batch.dtype).view(horizon + 1, 1, 1)
    interpolated = (h_grid / horizon) * y_batch.unsqueeze(0) + ((horizon - h_grid) / horizon) * x_batch.unsqueeze(0)
    x_h_minus_batch = [x.clone().detach().requires_grad_(True) for x in interpolated.unbind(0)]
    x_h_plus_batch = [x.clone().detach().requires_grad_(True) for x in interpolated.unbind(0)]
    return x_h_minus_batch, x_h_plus_batch


def run_primal_updates(
    model: torch.nn.Module,
    x_h_minus_batch: list[torch.Tensor],
    x_h_plus_batch: list[torch.Tensor],
    *,
    horizon: int,
    h_minus: torch.Tensor,
    h_plus: torch.Tensor,
    step_size: float,
    noise_scale: float,
    num_steps: int,
    time_chunk_size: int | None = None,
    clamp: tuple[float, float] | None = None,
):
    """Noisy Wasserstein gradient descent on the particles (Eq. 7). If `clamp` is given,
    particles are clamped to [clamp[0], clamp[1]] after every update."""
    batch_size = x_h_minus_batch[0].shape[0]
    num_time_steps = horizon + 1

    if time_chunk_size is not None and time_chunk_size <= 0:
        raise ValueError("time_chunk_size must be positive")
    use_full_horizon = time_chunk_size is None or time_chunk_size >= num_time_steps

    for _ in range(num_steps):
        model.zero_grad()

        if use_full_horizon:
            x_minus_cat = torch.cat(x_h_minus_batch, dim=0)
            x_plus_cat = torch.cat(x_h_plus_batch, dim=0)

            f_val_minus_output = model(torch.cat([x_minus_cat, h_minus], dim=-1))
            f_val_plus_output = model(torch.cat([x_plus_cat, h_plus], dim=-1))

            nabla_x_c_cat = grad_cost_x(x_minus_cat, x_plus_cat)
            nabla_y_c_cat = grad_cost_y(x_minus_cat, x_plus_cat)
            grads = torch.autograd.grad(
                outputs=(f_val_minus_output.sum(), f_val_plus_output.sum()),
                inputs=tuple(x_h_minus_batch) + tuple(x_h_plus_batch),
                create_graph=False,
            )

            nabla_x_f_minus = torch.stack(grads[:num_time_steps], dim=0)
            nabla_x_f_plus = torch.stack(grads[num_time_steps:], dim=0)
            nabla_x_c = nabla_x_c_cat.reshape(num_time_steps, batch_size, -1)
            nabla_y_c = nabla_y_c_cat.reshape(num_time_steps, batch_size, -1)

            delta_minus = nabla_x_c - nabla_x_f_minus
            delta_plus = nabla_y_c + nabla_x_f_plus

            with torch.no_grad():
                update_minus = step_size * delta_minus + torch.randn_like(delta_minus) * noise_scale
                update_plus = step_size * delta_plus + torch.randn_like(delta_plus) * noise_scale
                torch._foreach_sub_(x_h_minus_batch, list(update_minus.unbind(0)))
                torch._foreach_sub_(x_h_plus_batch, list(update_plus.unbind(0)))
                if clamp is not None:
                    _clamp_particles(x_h_minus_batch, x_h_plus_batch, clamp)
            continue

        state_shape = (num_time_steps, batch_size, x_h_minus_batch[0].shape[-1])
        noise_minus = torch.randn(state_shape, device=x_h_minus_batch[0].device, dtype=x_h_minus_batch[0].dtype)
        noise_plus = torch.randn(state_shape, device=x_h_plus_batch[0].device, dtype=x_h_plus_batch[0].dtype)

        for start in range(0, num_time_steps, time_chunk_size):
            end = min(start + time_chunk_size, num_time_steps)
            minus_chunk = x_h_minus_batch[start:end]
            plus_chunk = x_h_plus_batch[start:end]
            x_minus_cat = torch.cat(minus_chunk, dim=0)
            x_plus_cat = torch.cat(plus_chunk, dim=0)
            h_start = start * batch_size
            h_end = end * batch_size

            f_val_minus_output = model(torch.cat([x_minus_cat, h_minus[h_start:h_end]], dim=-1))
            f_val_plus_output = model(torch.cat([x_plus_cat, h_plus[h_start:h_end]], dim=-1))

            nabla_x_c_cat = grad_cost_x(x_minus_cat, x_plus_cat)
            nabla_y_c_cat = grad_cost_y(x_minus_cat, x_plus_cat)
            grads = torch.autograd.grad(
                outputs=(f_val_minus_output.sum(), f_val_plus_output.sum()),
                inputs=tuple(minus_chunk) + tuple(plus_chunk),
                create_graph=False,
            )

            chunk_steps = end - start
            nabla_x_f_minus = torch.stack(grads[:chunk_steps], dim=0)
            nabla_x_f_plus = torch.stack(grads[chunk_steps:], dim=0)
            nabla_x_c = nabla_x_c_cat.reshape(chunk_steps, batch_size, -1)
            nabla_y_c = nabla_y_c_cat.reshape(chunk_steps, batch_size, -1)

            delta_minus = nabla_x_c - nabla_x_f_minus
            delta_plus = nabla_y_c + nabla_x_f_plus

            with torch.no_grad():
                update_minus = step_size * delta_minus + noise_minus[start:end] * noise_scale
                update_plus = step_size * delta_plus + noise_plus[start:end] * noise_scale
                torch._foreach_sub_(minus_chunk, list(update_minus.unbind(0)))
                torch._foreach_sub_(plus_chunk, list(update_plus.unbind(0)))
                if clamp is not None:
                    _clamp_particles(minus_chunk, plus_chunk, clamp)

    return x_h_minus_batch, x_h_plus_batch


def _clamp_particles(
    x_h_minus_batch: list[torch.Tensor],
    x_h_plus_batch: list[torch.Tensor],
    clamp: tuple[float, float],
):
    low, high = float(clamp[0]), float(clamp[1])
    torch._foreach_clamp_min_(x_h_minus_batch, low)
    torch._foreach_clamp_max_(x_h_minus_batch, high)
    torch._foreach_clamp_min_(x_h_plus_batch, low)
    torch._foreach_clamp_max_(x_h_plus_batch, high)
