from __future__ import annotations

import torch


def cost_function(x: torch.Tensor, y: torch.Tensor):
    return 0.5 * torch.sum((x - y) ** 2, dim=-1)


def grad_cost_x(x: torch.Tensor, y: torch.Tensor):
    return x - y


def grad_cost_y(x: torch.Tensor, y: torch.Tensor):
    return -(x - y)


def compute_dual_loss(
    model: torch.nn.Module,
    x_star_batch: torch.Tensor,
    y_star_batch: torch.Tensor,
    x_h_minus_dual: list[torch.Tensor],
    x_h_plus_dual: list[torch.Tensor],
    h_steps: int,
    h_feat: torch.Tensor,
):
    batch_size = x_star_batch.shape[0]
    device = x_star_batch.device

    h_tensor_0 = torch.zeros((batch_size, 1), device=device, dtype=x_star_batch.dtype)
    loss_0 = (
        model(torch.cat([x_h_minus_dual[0], h_tensor_0], dim=-1)).sum()
        - model(torch.cat([x_star_batch.detach(), h_tensor_0], dim=-1)).sum()
    ) / batch_size

    x_plus_cat = torch.cat(x_h_plus_dual[:h_steps], dim=0)
    x_minus_cat = torch.cat(x_h_minus_dual[1 : h_steps + 1], dim=0)
    f_plus = model(torch.cat([x_plus_cat, h_feat], dim=-1))
    f_minus = model(torch.cat([x_minus_cat, h_feat], dim=-1))
    loss_intermediate_sum = -(f_plus.sum() - f_minus.sum()) / batch_size

    h_tensor_end = torch.ones((batch_size, 1), device=device, dtype=y_star_batch.dtype)
    loss_end = (
        model(torch.cat([y_star_batch.detach(), h_tensor_end], dim=-1)).sum()
        - model(torch.cat([x_h_plus_dual[h_steps], h_tensor_end], dim=-1)).sum()
    ) / batch_size

    return loss_0 + loss_intermediate_sum + loss_end


def backward_dual_loss(
    model: torch.nn.Module,
    x_star_batch: torch.Tensor,
    y_star_batch: torch.Tensor,
    x_h_minus_dual: list[torch.Tensor],
    x_h_plus_dual: list[torch.Tensor],
    h_steps: int,
    h_feat: torch.Tensor,
    *,
    time_chunk_size: int | None = None,
):
    if time_chunk_size is None or time_chunk_size >= h_steps:
        total_dual_loss = compute_dual_loss(
            model,
            x_star_batch,
            y_star_batch,
            x_h_minus_dual,
            x_h_plus_dual,
            h_steps=h_steps,
            h_feat=h_feat,
        )
        loss_value = float(total_dual_loss.item())
        total_dual_loss.backward()
        return loss_value

    if time_chunk_size <= 0:
        raise ValueError("time_chunk_size must be positive")

    batch_size = x_star_batch.shape[0]
    device = x_star_batch.device

    h_tensor_0 = torch.zeros((batch_size, 1), device=device, dtype=x_star_batch.dtype)
    loss_0 = (
        model(torch.cat([x_h_minus_dual[0], h_tensor_0], dim=-1)).sum()
        - model(torch.cat([x_star_batch.detach(), h_tensor_0], dim=-1)).sum()
    ) / batch_size
    loss_value = loss_0.detach()
    loss_0.backward()

    for start in range(0, h_steps, time_chunk_size):
        end = min(start + time_chunk_size, h_steps)
        x_plus_cat = torch.cat(x_h_plus_dual[start:end], dim=0)
        x_minus_cat = torch.cat(x_h_minus_dual[start + 1 : end + 1], dim=0)
        h_start = start * batch_size
        h_end = end * batch_size
        f_plus = model(torch.cat([x_plus_cat, h_feat[h_start:h_end]], dim=-1))
        f_minus = model(torch.cat([x_minus_cat, h_feat[h_start:h_end]], dim=-1))
        loss_intermediate = -(f_plus.sum() - f_minus.sum()) / batch_size
        loss_value = loss_value + loss_intermediate.detach()
        loss_intermediate.backward()

    h_tensor_end = torch.ones((batch_size, 1), device=device, dtype=y_star_batch.dtype)
    loss_end = (
        model(torch.cat([y_star_batch.detach(), h_tensor_end], dim=-1)).sum()
        - model(torch.cat([x_h_plus_dual[h_steps], h_tensor_end], dim=-1)).sum()
    ) / batch_size
    loss_value = loss_value + loss_end.detach()
    loss_end.backward()

    return float(loss_value.item())
