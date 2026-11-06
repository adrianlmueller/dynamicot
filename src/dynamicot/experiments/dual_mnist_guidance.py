from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import datasets, transforms

from ..data import build_image_source_data
from ..duals.image_trainer import (
    DEFAULT_IMAGE_TRAINING_TIME_CHUNK_SIZE,
    _build_optimizer,
    _initialize_ema_state,
    _parameter_l2_norm,
    _update_ema_state,
)
from ..duals.objectives import grad_cost_x, grad_cost_y
from ..primals.particles import build_time_features, initialize_particle_pairs, ot_permutation
from ..utils import create_results_dir, load_yaml_config, resolve_device, resolve_dtype, save_experiment_summary, save_json, set_seed
from ..utils.io import save_checkpoint
from ..utils.progress import ProgressBar


@dataclass
class MnistGuidanceTrainingArtifacts:
    model: torch.nn.Module
    source_data: torch.Tensor
    target_data: torch.Tensor
    target_labels: torch.Tensor
    target_digit_labels: torch.Tensor
    target_original: torch.Tensor
    width: int
    input_channels: int
    original_width: int
    model_architecture: str
    digits: list[int]
    label_values: dict[int, float]
    unconditional_label: float
    dual_loss_history: list[float]
    parameter_norm_history: list[float]
    snapshots: list[dict[str, torch.Tensor]]
    checkpoint_paths: list[str]
    final_checkpoint_path: str
    ema_state_dict: dict[str, torch.Tensor] | None
    ema_checkpoint_path: str | None
    metrics_path: str
    results_dir: str


class TimeLabelConditionedCNN(nn.Module):
    def __init__(
        self,
        image_size: int = 28,
        input_channels: int = 1,
        conv_channels: list[int] | tuple[int, ...] = (16, 32),
        fc_hidden_dim: int = 64,
        output_dim: int = 1,
        time_embedding_dim: int = 32,
        label_embedding_dim: int = 32,
        use_pooling: bool = True,
        pooling_type: str = "max",
        activation_type: str = "relu",
        softplus_beta: float = 1.0,
        final_activation: str = "sigmoid",
    ):
        super().__init__()
        self.image_size = image_size
        self.input_channels = input_channels
        self.use_pooling = use_pooling
        self.pooling_type = pooling_type
        self.activation_type = activation_type
        self.softplus_beta = softplus_beta
        self.final_activation = final_activation

        if pooling_type not in {"avg", "max"}:
            raise ValueError(f"Unsupported pooling_type: {pooling_type}")
        if activation_type == "softplus":
            activation = lambda: nn.Softplus(beta=softplus_beta)
        elif activation_type == "relu":
            activation = nn.ReLU
        else:
            raise ValueError(f"Unsupported activation_type: {activation_type}")

        self.time_embedding_net = nn.Sequential(
            nn.Linear(1, time_embedding_dim),
            activation(),
            nn.Linear(time_embedding_dim, time_embedding_dim),
            activation(),
        )
        self.label_embedding_net = nn.Sequential(
            nn.Linear(1, label_embedding_dim),
            activation(),
            nn.Linear(label_embedding_dim, label_embedding_dim),
            activation(),
        )

        self.conv_layers = nn.ModuleList()
        self.time_to_conv = nn.ModuleList()
        self.label_to_conv = nn.ModuleList()

        in_channels = input_channels
        spatial_size = image_size
        for out_channels in conv_channels:
            self.conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            self.time_to_conv.append(nn.Linear(time_embedding_dim, out_channels))
            self.label_to_conv.append(nn.Linear(label_embedding_dim, out_channels))
            in_channels = out_channels
            if use_pooling:
                spatial_size //= 2

        flattened_dim = in_channels * spatial_size * spatial_size
        self.fc_time = nn.Linear(time_embedding_dim, fc_hidden_dim)
        self.fc_label = nn.Linear(label_embedding_dim, fc_hidden_dim)
        self.fc1 = nn.Linear(flattened_dim, fc_hidden_dim)
        self.fc2 = nn.Linear(fc_hidden_dim, output_dim)
        self.activation = activation()
        self.pool = nn.AvgPool2d(2) if pooling_type == "avg" else nn.MaxPool2d(2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_and_h_and_label_normalized: torch.Tensor):
        x_flat = x_and_h_and_label_normalized[:, :-2]
        h_normalized = x_and_h_and_label_normalized[:, -2:-1]
        label_normalized = x_and_h_and_label_normalized[:, -1:].view(-1, 1)

        x = x_flat.view(-1, self.input_channels, self.image_size, self.image_size)
        time_embedding = self.time_embedding_net(h_normalized)
        label_embedding = self.label_embedding_net(label_normalized)

        for conv, time_projection, label_projection in zip(self.conv_layers, self.time_to_conv, self.label_to_conv):
            x = conv(x)
            x = x + time_projection(time_embedding).unsqueeze(-1).unsqueeze(-1)
            x = x + label_projection(label_embedding).unsqueeze(-1).unsqueeze(-1)
            x = self.activation(x)
            if self.use_pooling:
                x = self.pool(x)

        x = x.flatten(start_dim=1)
        x = self.fc1(x) + self.fc_time(time_embedding) + self.fc_label(label_embedding)
        x = self.activation(x)
        x = self.fc2(x)

        if self.final_activation == "sigmoid":
            return self.sigmoid(x)
        if self.final_activation == "identity":
            return x
        raise ValueError(f"Unsupported final_activation: {self.final_activation}")


def build_time_label_conditioned_image_model(
    model_cfg: dict[str, Any],
    *,
    image_size: int,
    input_channels: int,
):
    architecture = str(model_cfg.get("architecture", "cnn")).lower()
    if architecture != "cnn":
        raise ValueError("dual_mnist_guidance currently supports model.architecture: cnn")

    time_embedding_dim = int(model_cfg.get("time_embedding_dim", 32))
    return TimeLabelConditionedCNN(
        image_size=image_size,
        input_channels=input_channels,
        conv_channels=list(model_cfg["conv_channels"]),
        fc_hidden_dim=int(model_cfg["fc_hidden_dim"]),
        output_dim=int(model_cfg.get("output_dim", 1)),
        time_embedding_dim=time_embedding_dim,
        label_embedding_dim=int(model_cfg.get("label_embedding_dim", time_embedding_dim)),
        use_pooling=bool(model_cfg.get("use_pooling", True)),
        pooling_type=str(model_cfg.get("pooling_type", "max")),
        activation_type=str(model_cfg.get("activation_type", "relu")),
        softplus_beta=float(model_cfg.get("softplus_beta", 1.0)),
        final_activation=str(model_cfg.get("final_activation", "sigmoid")),
    )


def _resolve_digits(data_cfg: dict[str, Any]):
    digits = sorted(set(int(digit) for digit in data_cfg.get("digits", [0, 1])))
    if not digits:
        raise ValueError("data.digits must contain at least one MNIST digit")
    if digits[0] < 0 or digits[-1] > 9:
        raise ValueError("MNIST digits must be in the range 0..9")
    return digits


def _select_mnist_targets_with_labels(config: dict[str, Any], *, device: torch.device, dtype: torch.dtype):
    root = config.get("root", "./data")
    train = bool(config.get("train", True))
    download = bool(config.get("download", True))
    digits = _resolve_digits(config)
    n_samples = int(config["n_samples"])

    dataset = datasets.MNIST(
        root=root,
        train=train,
        download=download,
        transform=transforms.ToTensor(),
    )

    mask = torch.zeros_like(dataset.targets, dtype=torch.bool)
    for digit in digits:
        mask = mask | (dataset.targets == digit)
    indices = mask.nonzero(as_tuple=True)[0]

    mnist_data = dataset.data[indices].to(dtype=torch.float32) / 255.0
    mnist_digit_labels = dataset.targets[indices].to(dtype=torch.long)
    if n_samples > mnist_data.shape[0]:
        raise ValueError(f"Requested {n_samples} MNIST samples, but only {mnist_data.shape[0]} are available")

    perm = torch.randperm(mnist_data.shape[0])
    target_original = mnist_data[perm][:n_samples].to(device=device, dtype=dtype)
    target_digit_labels = mnist_digit_labels[perm][:n_samples].to(device=device)
    return target_original, target_digit_labels, digits


def _label_values_for_digits(digits: list[int]):
    num_classes = len(digits)
    return {digit: class_idx / num_classes for class_idx, digit in enumerate(digits)}


def _normalize_digit_labels(
    digit_labels: torch.Tensor,
    *,
    digits: list[int],
    dtype: torch.dtype,
):
    label_values = _label_values_for_digits(digits)
    labels = torch.empty((digit_labels.shape[0], 1), device=digit_labels.device, dtype=dtype)
    for digit, label_value in label_values.items():
        labels[digit_labels == digit] = float(label_value)
    return labels, label_values


def _unconditional_label_from_config(config: dict[str, Any]):
    training_cfg = config.get("training", {})
    conditioning_cfg = training_cfg.get("conditioning", {})
    if "unconditional_label" in conditioning_cfg:
        return float(conditioning_cfg["unconditional_label"])
    return float(config.get("model", {}).get("unconditional_label", 1.0))


def load_mnist_guidance_vectors(config: dict[str, Any], *, device: torch.device, dtype: torch.dtype):
    n_samples = int(config["n_samples"])
    width = int(config.get("width", 10))

    target_original, target_digit_labels, digits = _select_mnist_targets_with_labels(config, device=device, dtype=dtype)
    target_labels, label_values = _normalize_digit_labels(target_digit_labels, digits=digits, dtype=dtype)
    target_images = target_original.view(n_samples, 1, 28, 28)
    target_downsampled = F.adaptive_avg_pool2d(target_images, (width, width))
    target_data = target_downsampled.view(n_samples, width * width)
    source_data = build_image_source_data(
        config.get("source"),
        n_samples=n_samples,
        width=width,
        input_channels=1,
        device=device,
        dtype=dtype,
    )

    metadata = {
        "width": width,
        "original_width": 28,
        "target_original": target_original.view(n_samples, 28 * 28),
        "target_labels": target_labels,
        "target_digit_labels": target_digit_labels,
        "digits": digits,
        "label_values": label_values,
    }
    return source_data, target_data, metadata


def prepare_mnist_guidance_experiment(config: dict[str, Any]):
    seed = config.get("seed")
    if seed is not None:
        set_seed(int(seed))

    device = resolve_device(config.get("device", "cpu"))
    dtype = resolve_dtype(config.get("dtype", "float32"))
    source_data, target_data, metadata = load_mnist_guidance_vectors(config["data"], device=device, dtype=dtype)

    model_cfg = config["model"]
    width = int(metadata["width"])
    input_channels = int(model_cfg.get("input_channels", 1))
    model = build_time_label_conditioned_image_model(
        model_cfg,
        image_size=width,
        input_channels=input_channels,
    ).to(device)

    metadata["unconditional_label"] = _unconditional_label_from_config(config)
    return model, source_data, target_data, metadata


def _sample_training_batch_with_labels(
    source_data: torch.Tensor,
    target_data: torch.Tensor,
    target_labels: torch.Tensor,
    *,
    step: int,
    batch_size: int,
    batching_type: str,
    state: dict[str, torch.Tensor] | None = None,
):
    if batching_type not in {"random", "paired", "fixed"}:
        raise ValueError("batching_type must be one of: random, paired, fixed")

    state = {} if state is None else dict(state)
    n_samples = source_data.shape[0]
    batches_per_epoch = n_samples // batch_size
    batch_in_epoch = step % batches_per_epoch
    device = source_data.device

    if batching_type == "random":
        if batch_in_epoch == 0 or "source_shuffled" not in state:
            source_perm = torch.randperm(n_samples, device=device)
            target_perm = torch.randperm(n_samples, device=device)
            state["source_shuffled"] = source_data[source_perm]
            state["target_shuffled"] = target_data[target_perm]
            state["label_shuffled"] = target_labels[target_perm]
        start = batch_in_epoch * batch_size
        end = start + batch_size
        return (
            state["source_shuffled"][start:end].detach().clone(),
            state["target_shuffled"][start:end].detach().clone(),
            state["label_shuffled"][start:end].detach().clone(),
            state,
        )

    if batching_type == "paired":
        if batch_in_epoch == 0 or "source_shuffled" not in state:
            perm = torch.randperm(n_samples, device=device)
            state["source_shuffled"] = source_data[perm]
            state["target_shuffled"] = target_data[perm]
            state["label_shuffled"] = target_labels[perm]
        start = batch_in_epoch * batch_size
        end = start + batch_size
        return (
            state["source_shuffled"][start:end].detach().clone(),
            state["target_shuffled"][start:end].detach().clone(),
            state["label_shuffled"][start:end].detach().clone(),
            state,
        )

    if batch_in_epoch == 0 or "shuffled_batch_numbers" not in state:
        state["shuffled_batch_numbers"] = torch.randperm(batches_per_epoch, device=device)
    batch_number = int(state["shuffled_batch_numbers"][batch_in_epoch].item())
    start = batch_number * batch_size
    end = start + batch_size
    return (
        source_data[start:end].detach().clone(),
        target_data[start:end].detach().clone(),
        target_labels[start:end].detach().clone(),
        state,
    )


def _apply_initialization_coupling_with_labels(
    x_batch: torch.Tensor,
    y_batch: torch.Tensor,
    labels: torch.Tensor,
    *,
    initialization_type: str,
):
    initialization_type = initialization_type.lower()
    if initialization_type == "linear":
        return x_batch, y_batch, labels
    if initialization_type == "ot":
        perm = ot_permutation(x_batch, y_batch, p=2.0)
        return x_batch, y_batch[perm].detach().clone(), labels[perm].detach().clone()
    raise ValueError("initialization_type must be one of: linear, ot")


def _run_conditional_primal_updates(
    model: torch.nn.Module,
    x_h_minus_batch: list[torch.Tensor],
    x_h_plus_batch: list[torch.Tensor],
    labels: torch.Tensor,
    *,
    horizon: int,
    h_minus: torch.Tensor,
    h_plus: torch.Tensor,
    step_size: float,
    noise_scale: float,
    num_steps: int,
    time_chunk_size: int | None = None,
):
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
            label_input = labels.repeat(num_time_steps, 1)

            f_val_minus_output = model(torch.cat([x_minus_cat, h_minus, label_input], dim=-1))
            f_val_plus_output = model(torch.cat([x_plus_cat, h_plus, label_input], dim=-1))

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
            label_input = labels.repeat(end - start, 1)

            f_val_minus_output = model(torch.cat([x_minus_cat, h_minus[h_start:h_end], label_input], dim=-1))
            f_val_plus_output = model(torch.cat([x_plus_cat, h_plus[h_start:h_end], label_input], dim=-1))

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

    return x_h_minus_batch, x_h_plus_batch


def _compute_conditional_dual_loss(
    model: torch.nn.Module,
    x_star_batch: torch.Tensor,
    y_star_batch: torch.Tensor,
    labels: torch.Tensor,
    x_h_minus_dual: list[torch.Tensor],
    x_h_plus_dual: list[torch.Tensor],
    h_steps: int,
    h_feat: torch.Tensor,
):
    batch_size = x_star_batch.shape[0]
    device = x_star_batch.device

    h_tensor_0 = torch.zeros((batch_size, 1), device=device, dtype=x_star_batch.dtype)
    loss_0 = (
        model(torch.cat([x_h_minus_dual[0], h_tensor_0, labels], dim=-1)).sum()
        - model(torch.cat([x_star_batch.detach(), h_tensor_0, labels], dim=-1)).sum()
    ) / batch_size

    x_plus_cat = torch.cat(x_h_plus_dual[:h_steps], dim=0)
    x_minus_cat = torch.cat(x_h_minus_dual[1 : h_steps + 1], dim=0)
    label_feat = labels.repeat(h_steps, 1)
    f_plus = model(torch.cat([x_plus_cat, h_feat, label_feat], dim=-1))
    f_minus = model(torch.cat([x_minus_cat, h_feat, label_feat], dim=-1))
    loss_intermediate_sum = -(f_plus.sum() - f_minus.sum()) / batch_size

    h_tensor_end = torch.ones((batch_size, 1), device=device, dtype=y_star_batch.dtype)
    loss_end = (
        model(torch.cat([y_star_batch.detach(), h_tensor_end, labels], dim=-1)).sum()
        - model(torch.cat([x_h_plus_dual[h_steps], h_tensor_end, labels], dim=-1)).sum()
    ) / batch_size

    return loss_0 + loss_intermediate_sum + loss_end


def _backward_conditional_dual_loss(
    model: torch.nn.Module,
    x_star_batch: torch.Tensor,
    y_star_batch: torch.Tensor,
    labels: torch.Tensor,
    x_h_minus_dual: list[torch.Tensor],
    x_h_plus_dual: list[torch.Tensor],
    h_steps: int,
    h_feat: torch.Tensor,
    *,
    time_chunk_size: int | None = None,
):
    if time_chunk_size is None or time_chunk_size >= h_steps:
        total_dual_loss = _compute_conditional_dual_loss(
            model,
            x_star_batch,
            y_star_batch,
            labels,
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
        model(torch.cat([x_h_minus_dual[0], h_tensor_0, labels], dim=-1)).sum()
        - model(torch.cat([x_star_batch.detach(), h_tensor_0, labels], dim=-1)).sum()
    ) / batch_size
    loss_value = loss_0.detach()
    loss_0.backward()

    for start in range(0, h_steps, time_chunk_size):
        end = min(start + time_chunk_size, h_steps)
        x_plus_cat = torch.cat(x_h_plus_dual[start:end], dim=0)
        x_minus_cat = torch.cat(x_h_minus_dual[start + 1 : end + 1], dim=0)
        h_start = start * batch_size
        h_end = end * batch_size
        label_feat = labels.repeat(end - start, 1)
        f_plus = model(torch.cat([x_plus_cat, h_feat[h_start:h_end], label_feat], dim=-1))
        f_minus = model(torch.cat([x_minus_cat, h_feat[h_start:h_end], label_feat], dim=-1))
        loss_intermediate = -(f_plus.sum() - f_minus.sum()) / batch_size
        loss_value = loss_value + loss_intermediate.detach()
        loss_intermediate.backward()

    h_tensor_end = torch.ones((batch_size, 1), device=device, dtype=y_star_batch.dtype)
    loss_end = (
        model(torch.cat([y_star_batch.detach(), h_tensor_end, labels], dim=-1)).sum()
        - model(torch.cat([x_h_plus_dual[h_steps], h_tensor_end, labels], dim=-1)).sum()
    ) / batch_size
    loss_value = loss_value + loss_end.detach()
    loss_end.backward()

    return float(loss_value.item())


def generate_from_source_guided(
    model: torch.nn.Module,
    source_data: torch.Tensor,
    horizon: int,
    *,
    input_labels: torch.Tensor | None = None,
    unconditional_label: float = 1.0,
    guidance_scale: float = 0.0,
    noise_scale: float = 0.0,
    clamp: tuple[float, float] | None = None,
):
    num_points = source_data.shape[0]
    if input_labels is None:
        input_labels = torch.full(
            (num_points, 1),
            float(unconditional_label),
            device=source_data.device,
            dtype=source_data.dtype,
        )
    else:
        input_labels = input_labels.to(device=source_data.device, dtype=source_data.dtype)

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
        model_output = model(torch.cat([current, h_tensor, input_labels], dim=-1))
        gradient = torch.autograd.grad(model_output.sum(), current, create_graph=False)[0]

        if guidance_scale > 1e-8:
            unconditional_labels = torch.full_like(input_labels, float(unconditional_label))
            unconditional_output = model(torch.cat([current, h_tensor, unconditional_labels], dim=-1))
            unconditional_gradient = torch.autograd.grad(
                unconditional_output.sum(),
                current,
                create_graph=False,
            )[0]
            gradient = (1.0 + guidance_scale) * gradient - guidance_scale * unconditional_gradient

        if noise_scale > 1e-8:
            next_state = current.detach() - gradient.detach() + noise_scale * torch.randn_like(current)
        else:
            next_state = current.detach() - gradient.detach()
        if clamp is not None:
            next_state = next_state.clamp(*clamp)
        generated_path.append(next_state.requires_grad_(True))

    model.train(was_training)
    return generated_path


def train_mnist_guidance_dual_model(config: dict[str, Any]):
    model, source_data, target_data, metadata = prepare_mnist_guidance_experiment(config)
    target_labels = metadata["target_labels"]
    optimizer = _build_optimizer(model, config)
    results_dir = create_results_dir(config)
    save_experiment_summary(results_dir, config)

    training_cfg = config["training"]
    conditioning_cfg = training_cfg.get("conditioning", {})
    horizon = int(training_cfg["horizon"])
    iterations = int(training_cfg["iterations"])
    snapshot_interval = int(training_cfg["snapshot_interval"])
    batch_size = int(training_cfg["batching"]["batch_size"])
    time_chunk_size = int(training_cfg.get("time_chunk_size", DEFAULT_IMAGE_TRAINING_TIME_CHUNK_SIZE))
    initialization_type = str(training_cfg.get("initialization_type", "linear")).lower()
    p_unconditional = float(conditioning_cfg.get("p_unconditional", training_cfg.get("p_unconditional", 0.0)))
    unconditional_label = _unconditional_label_from_config(config)
    if not 0.0 <= p_unconditional <= 1.0:
        raise ValueError("training.conditioning.p_unconditional must be between 0 and 1")

    ema_decay = training_cfg.get("ema_decay")
    ema_decay = None if ema_decay is None else float(ema_decay)
    if ema_decay is not None and not 0.0 < ema_decay < 1.0:
        raise ValueError("ema_decay must be between 0 and 1")

    if source_data.shape[0] % batch_size != 0:
        raise ValueError("batch_size must divide n_samples")

    h_minus, h_plus, h_feat = build_time_features(
        horizon=horizon,
        batch_size=batch_size,
        device=source_data.device,
        dtype=source_data.dtype,
    )

    dual_loss_history: list[float] = []
    parameter_norm_history: list[float] = []
    snapshots: list[dict[str, torch.Tensor]] = []
    checkpoint_paths: list[str] = []
    batch_state: dict[str, torch.Tensor] = {}
    ema_state = _initialize_ema_state(model) if ema_decay is not None else None

    progress = ProgressBar(iterations, desc="mnist guidance training", unit="iter")
    for step in progress:
        x_batch, y_batch, label_batch, batch_state = _sample_training_batch_with_labels(
            source_data,
            target_data,
            target_labels,
            step=step,
            batch_size=batch_size,
            batching_type=training_cfg["batching"]["type"],
            state=batch_state,
        )
        x_batch, y_batch, label_batch = _apply_initialization_coupling_with_labels(
            x_batch,
            y_batch,
            label_batch,
            initialization_type=initialization_type,
        )

        model_labels = label_batch
        if p_unconditional > 0.0:
            model_labels = label_batch.clone()
            unconditional_mask = torch.rand(batch_size, device=model_labels.device) < p_unconditional
            model_labels[unconditional_mask] = unconditional_label

        x_h_minus_batch, x_h_plus_batch = initialize_particle_pairs(
            x_batch,
            y_batch,
            horizon=horizon,
        )

        x_h_minus_batch, x_h_plus_batch = _run_conditional_primal_updates(
            model,
            x_h_minus_batch,
            x_h_plus_batch,
            model_labels,
            horizon=horizon,
            h_minus=h_minus,
            h_plus=h_plus,
            step_size=float(training_cfg["primal"]["lr"]),
            noise_scale=float(training_cfg["primal"]["noise_scale"]),
            num_steps=int(training_cfg["primal"]["num_steps"]),
            time_chunk_size=time_chunk_size,
        )

        optimizer.zero_grad(set_to_none=True)
        x_h_minus_dual = [state.detach() for state in x_h_minus_batch]
        x_h_plus_dual = [state.detach() for state in x_h_plus_batch]

        dual_loss = _backward_conditional_dual_loss(
            model,
            x_batch,
            y_batch,
            model_labels,
            x_h_minus_dual,
            x_h_plus_dual,
            h_steps=horizon,
            h_feat=h_feat,
            time_chunk_size=time_chunk_size,
        )

        dual_loss_history.append(dual_loss)
        optimizer.step()
        if ema_state is not None:
            _update_ema_state(ema_state, model, ema_decay)

        parameter_norm_history.append(_parameter_l2_norm(model))
        progress.set_postfix(
            loss=f"{dual_loss_history[-1]:.4f}",
            norm=f"{parameter_norm_history[-1]:.2f}",
        )

        if (step + 1) % snapshot_interval == 0:
            snapshots.append(copy.deepcopy(model.state_dict()))
            checkpoint_paths.append(
                save_checkpoint(
                    results_dir,
                    filename=f"checkpoint_step_{step + 1:07d}.pt",
                    model=model,
                    optimizer=optimizer,
                    step=step + 1,
                )
            )
            progress.write(f"Saved checkpoint at step {step + 1} -> {checkpoint_paths[-1]}")

    progress.close()

    metrics_path = save_json(
        results_dir,
        "metrics.json",
        {
            "dual_loss_history": dual_loss_history,
            "parameter_norm_history": parameter_norm_history,
            "snapshot_interval": snapshot_interval,
            "checkpoint_paths": checkpoint_paths,
            "ema_decay": ema_decay,
            "p_unconditional": p_unconditional,
            "unconditional_label": unconditional_label,
        },
    )

    final_checkpoint_path = save_checkpoint(
        results_dir,
        filename="checkpoint_final.pt",
        model=model,
        optimizer=optimizer,
        step=iterations,
    )
    ema_checkpoint_path = None
    if ema_state is not None:
        ema_checkpoint_path = save_checkpoint(
            results_dir,
            filename="checkpoint_final_ema.pt",
            model=None,
            model_state_dict=ema_state,
            optimizer=None,
            step=iterations,
        )

    return MnistGuidanceTrainingArtifacts(
        model=model,
        source_data=source_data,
        target_data=target_data,
        target_labels=target_labels,
        target_digit_labels=metadata["target_digit_labels"],
        target_original=metadata["target_original"],
        width=int(metadata["width"]),
        input_channels=int(config["model"].get("input_channels", 1)),
        original_width=int(metadata.get("original_width", metadata["width"])),
        model_architecture=str(config["model"].get("architecture", "cnn")).lower(),
        digits=list(metadata["digits"]),
        label_values=dict(metadata["label_values"]),
        unconditional_label=unconditional_label,
        dual_loss_history=dual_loss_history,
        parameter_norm_history=parameter_norm_history,
        snapshots=snapshots,
        checkpoint_paths=checkpoint_paths,
        final_checkpoint_path=final_checkpoint_path,
        ema_state_dict=ema_state,
        ema_checkpoint_path=ema_checkpoint_path,
        metrics_path=metrics_path,
        results_dir=results_dir,
    )


def main():
    parser = argparse.ArgumentParser(description="Train the conditional MNIST guidance dual dynamic OT experiment.")
    parser.add_argument(
        "--config",
        default="configs/dual_mnist_guidance.yaml",
        help="Path to a YAML experiment config.",
    )
    parser.add_argument(
        "--num-eval-samples-per-class",
        type=int,
        default=None,
        help="Number of examples per MNIST digit to evaluate and plot after training.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=None,
        help="Classifier-free guidance scale for the guided evaluation plot.",
    )
    args = parser.parse_args()

    config = load_yaml_config(args.config)

    from .dual_mnist_guidance_eval import evaluate_mnist_guidance_checkpoint
    from .dual_mnist_guidance_plot import plot_mnist_guidance_evaluation

    artifacts = train_mnist_guidance_dual_model(config)
    print(f"Training completed. Results dir: {artifacts.results_dir}")
    print(f"Stored {len(artifacts.checkpoint_paths)} checkpoint files.")
    print(f"Final checkpoint: {artifacts.final_checkpoint_path}")
    if artifacts.ema_checkpoint_path is not None:
        print(f"Final EMA checkpoint: {artifacts.ema_checkpoint_path}")
    print(f"Metrics file: {artifacts.metrics_path}")

    evaluation_path = evaluate_mnist_guidance_checkpoint(
        artifacts.results_dir,
        num_samples_per_class=args.num_eval_samples_per_class,
        guidance_scale=args.guidance_scale,
    )
    print(f"Saved evaluation artifact: {evaluation_path}")
    for plot_path in plot_mnist_guidance_evaluation(artifacts.results_dir):
        print(f"Saved plot: {plot_path}")

    if artifacts.ema_checkpoint_path is not None:
        ema_evaluation_path = evaluate_mnist_guidance_checkpoint(
            artifacts.results_dir,
            checkpoint_filename="checkpoint_final_ema.pt",
            evaluation_filename="evaluation_ema.pt",
            num_samples_per_class=args.num_eval_samples_per_class,
            guidance_scale=args.guidance_scale,
        )
        print(f"Saved EMA evaluation artifact: {ema_evaluation_path}")
        for plot_path in plot_mnist_guidance_evaluation(
            artifacts.results_dir,
            evaluation_filename="evaluation_ema.pt",
            filename_suffix="ema",
            skip_metrics=True,
        ):
            print(f"Saved EMA plot: {plot_path}")


if __name__ == "__main__":
    main()
