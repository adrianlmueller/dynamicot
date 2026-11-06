from __future__ import annotations

import torch


def generate_from_source(
    model: torch.nn.Module,
    source_data: torch.Tensor,
    horizon: int,
    noise_scale: float = 0.0,
    clamp: tuple[float, float] | None = None,
):
    num_points = source_data.shape[0]
    generated_path = [source_data.clone().detach().requires_grad_(True)]
    was_training = model.training
    model.eval()

    for h_idx in range(horizon + 1):
        current = generated_path[h_idx]
        h_tensor = torch.full(
            (num_points, 1),
            h_idx / (horizon + 1),
            device=current.device,
            dtype=current.dtype,
        )
        mlp_output = model(torch.cat([current, h_tensor], dim=-1))
        gradient = torch.autograd.grad(mlp_output.sum(), current, create_graph=False)[0]
        if noise_scale > 1e-8:
            next_state = current.detach() - gradient.detach() + noise_scale * torch.randn_like(current)
        else:
            next_state = current.detach() - gradient.detach()
        if clamp is not None:
            next_state = next_state.clamp(*clamp)
        generated_path.append(next_state.requires_grad_(True))

    model.train(was_training)
    return generated_path


def generate_from_target(
    model: torch.nn.Module,
    target_data: torch.Tensor,
    horizon: int,
    noise_scale: float = 0.0,
    clamp: tuple[float, float] | None = None,
):
    num_points = target_data.shape[0]
    generated_path = [target_data.clone().detach().requires_grad_(True)]
    was_training = model.training
    model.eval()

    for h_idx in range(horizon + 1):
        current = generated_path[h_idx]
        actual_mlp_h_index = horizon + 1 - h_idx
        h_tensor = torch.full(
            (num_points, 1),
            actual_mlp_h_index / (horizon + 1),
            device=current.device,
            dtype=current.dtype,
        )
        mlp_output = model(torch.cat([current, h_tensor], dim=-1))
        gradient = torch.autograd.grad(mlp_output.sum(), current, create_graph=False)[0]
        if noise_scale > 1e-8:
            next_state = current.detach() + gradient.detach() + noise_scale * torch.randn_like(current)
        else:
            next_state = current.detach() + gradient.detach()
        if clamp is not None:
            next_state = next_state.clamp(*clamp)
        generated_path.append(next_state.requires_grad_(True))

    model.train(was_training)
    return generated_path


def generate_from_source_few_steps(
    model: torch.nn.Module,
    source_data: torch.Tensor,
    horizon: int,
    num_steps: int,
    noise_scale: float = 0.0,
    clamp: tuple[float, float] | None = None,
):
    # Each of the num_steps updates covers (horizon + 1) // num_steps training steps; if
    # num_steps does not divide horizon + 1, the remaining steps are skipped.

    num_points = source_data.shape[0]
    step_size = (horizon + 1) // num_steps
    generated_path = [source_data.clone().detach().requires_grad_(True)]
    was_training = model.training
    model.eval()

    for step_idx in range(num_steps):
        current = generated_path[step_idx]
        h_idx = step_idx * step_size
        h_tensor = torch.full(
            (num_points, 1),
            h_idx / (horizon + 1),
            device=current.device,
            dtype=current.dtype,
        )
        model_output = model(torch.cat([current, h_tensor], dim=-1))
        gradient = torch.autograd.grad(model_output.sum(), current, create_graph=False)[0]

        if noise_scale > 1e-8:
            next_state = current.detach() + step_size * (-gradient.detach() + noise_scale * torch.randn_like(current))
        else:
            next_state = current.detach() - step_size * gradient.detach()
        if clamp is not None:
            next_state = next_state.clamp(*clamp)
        generated_path.append(next_state.requires_grad_(True))

    model.train(was_training)
    return generated_path


def generate_from_target_few_steps(
    model: torch.nn.Module,
    target_data: torch.Tensor,
    horizon: int,
    num_steps: int,
    noise_scale: float = 0.0,
    clamp: tuple[float, float] | None = None,
):
    # Each of the num_steps updates covers (horizon + 1) // num_steps training steps; if
    # num_steps does not divide horizon + 1, the remaining steps are skipped.

    num_points = target_data.shape[0]
    step_size = (horizon + 1) // num_steps
    generated_path = [target_data.clone().detach().requires_grad_(True)]
    was_training = model.training
    model.eval()

    for step_idx in range(num_steps):
        current = generated_path[step_idx]
        actual_mlp_h_index = horizon + 1 - step_idx * step_size
        h_tensor = torch.full(
            (num_points, 1),
            actual_mlp_h_index / (horizon + 1),
            device=current.device,
            dtype=current.dtype,
        )
        model_output = model(torch.cat([current, h_tensor], dim=-1))
        gradient = torch.autograd.grad(model_output.sum(), current, create_graph=False)[0]

        if noise_scale > 1e-8:
            next_state = current.detach() + step_size * (gradient.detach() + noise_scale * torch.randn_like(current))
        else:
            next_state = current.detach() + step_size * gradient.detach()
        if clamp is not None:
            next_state = next_state.clamp(*clamp)
        generated_path.append(next_state.requires_grad_(True))

    model.train(was_training)
    return generated_path
