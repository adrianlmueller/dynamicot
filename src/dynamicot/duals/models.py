from __future__ import annotations

import math

import torch
from torch import nn


class TimeEmbeddingMLP(nn.Module):
    def __init__(
        self,
        spatial_dim: int,
        hidden_dims: list[int],
        output_dim: int = 1,
        time_embedding_dim: int = 32,
    ):
        super().__init__()
        self.spatial_dim = spatial_dim

        self.time_embedding_net = nn.Sequential(
            nn.Linear(1, time_embedding_dim),
            nn.ReLU(),
            nn.Linear(time_embedding_dim, time_embedding_dim),
            nn.ReLU(),
        )

        layers: list[nn.Module] = [
            nn.Linear(spatial_dim + time_embedding_dim, hidden_dims[0]),
            nn.ReLU(),
        ]
        for in_dim, out_dim in zip(hidden_dims, hidden_dims[1:]):
            layers.extend([nn.Linear(in_dim, out_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x_and_h_normalized: torch.Tensor):
        spatial_data = x_and_h_normalized[:, : self.spatial_dim]
        h_normalized = x_and_h_normalized[:, self.spatial_dim :].view(-1, 1)
        time_embedding = self.time_embedding_net(h_normalized)
        combined_input = torch.cat([spatial_data, time_embedding], dim=-1)
        return self.network(combined_input)


class TimeConditionedCNN(nn.Module):
    def __init__(
        self,
        image_size: int = 28,
        input_channels: int = 1,
        conv_channels: list[int] | tuple[int, ...] = (16, 32),
        fc_hidden_dim: int = 64,
        output_dim: int = 1,
        time_embedding_dim: int = 32,
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

        self.conv_layers = nn.ModuleList()
        self.time_to_conv = nn.ModuleList()

        in_channels = input_channels
        spatial_size = image_size
        for out_channels in conv_channels:
            self.conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            self.time_to_conv.append(nn.Linear(time_embedding_dim, out_channels))
            in_channels = out_channels
            if use_pooling:
                spatial_size //= 2

        flattened_dim = in_channels * spatial_size * spatial_size
        self.fc_time = nn.Linear(time_embedding_dim, fc_hidden_dim)
        self.fc1 = nn.Linear(flattened_dim, fc_hidden_dim)
        self.fc2 = nn.Linear(fc_hidden_dim, output_dim)
        self.activation = activation()
        self.pool = nn.AvgPool2d(2) if pooling_type == "avg" else nn.MaxPool2d(2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_and_h_normalized: torch.Tensor):
        x_flat = x_and_h_normalized[:, :-1]
        h_normalized = x_and_h_normalized[:, -1:].view(-1, 1)

        x = x_flat.view(-1, self.input_channels, self.image_size, self.image_size)
        time_embedding = self.time_embedding_net(h_normalized)

        for conv, time_projection in zip(self.conv_layers, self.time_to_conv):
            x = conv(x)
            x = x + time_projection(time_embedding).unsqueeze(-1).unsqueeze(-1)
            x = self.activation(x)
            if self.use_pooling:
                x = self.pool(x)

        x = x.flatten(start_dim=1)
        x = self.fc1(x) + self.fc_time(time_embedding)
        x = self.activation(x)
        x = self.fc2(x)

        if self.final_activation == "sigmoid":
            return self.sigmoid(x)
        if self.final_activation == "identity":
            return x
        raise ValueError(f"Unsupported final_activation: {self.final_activation}")


def _make_activation(activation_type: str, *, softplus_beta: float = 1.0):
    activation_type = activation_type.lower()
    if activation_type == "silu":
        return nn.SiLU()
    if activation_type == "gelu":
        return nn.GELU()
    if activation_type == "softplus":
        return nn.Softplus(beta=softplus_beta)
    if activation_type == "relu":
        return nn.ReLU()
    raise ValueError(f"Unsupported activation_type: {activation_type}")


def _resolve_group_count(num_channels: int, requested_groups: int):
    groups = max(1, min(int(requested_groups), num_channels))
    while num_channels % groups != 0:
        groups -= 1
    return groups


class FourierScalarEmbedding(nn.Module):
    def __init__(self, embedding_dim: int = 64):
        super().__init__()
        if embedding_dim % 2 != 0:
            raise ValueError("embedding_dim must be even for sin/cos Fourier features")
        frequencies = torch.arange(1, embedding_dim // 2 + 1, dtype=torch.float32)
        self.register_buffer("frequencies", frequencies, persistent=False)

    def forward(self, scalar: torch.Tensor):
        scalar = scalar.view(-1, 1)
        angles = 2.0 * math.pi * scalar * self.frequencies.view(1, -1)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class AdaGroupNorm(nn.Module):
    def __init__(self, num_channels: int, conditioning_dim: int, *, norm_groups: int = 8):
        super().__init__()
        self.norm = nn.GroupNorm(
            _resolve_group_count(num_channels, norm_groups),
            num_channels,
            affine=False,
        )
        self.conditioning_to_scale_shift = nn.Linear(conditioning_dim, 2 * num_channels)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor):
        scale, shift = self.conditioning_to_scale_shift(conditioning).chunk(2, dim=-1)
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)
        return (1.0 + scale) * self.norm(x) + shift


class ConditionedResidualBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        conditioning_dim: int,
        *,
        norm_groups: int = 8,
        activation_type: str = "silu",
        softplus_beta: float = 1.0,
    ):
        super().__init__()
        self.norm1 = AdaGroupNorm(channels, conditioning_dim, norm_groups=norm_groups)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = AdaGroupNorm(channels, conditioning_dim, norm_groups=norm_groups)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.activation = _make_activation(activation_type, softplus_beta=softplus_beta)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor):
        residual = self.activation(self.norm1(x, conditioning))
        residual = self.conv1(residual)
        residual = self.activation(self.norm2(residual, conditioning))
        residual = self.conv2(residual)
        return x + residual


class ConditionedDownsampleBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conditioning_dim: int,
        *,
        norm_groups: int = 8,
        activation_type: str = "silu",
        softplus_beta: float = 1.0,
    ):
        super().__init__()
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)
        self.norm1 = AdaGroupNorm(in_channels, conditioning_dim, norm_groups=norm_groups)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.norm2 = AdaGroupNorm(out_channels, conditioning_dim, norm_groups=norm_groups)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.activation = _make_activation(activation_type, softplus_beta=softplus_beta)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor):
        residual = self.activation(self.norm1(x, conditioning))
        residual = self.conv1(residual)
        residual = self.activation(self.norm2(residual, conditioning))
        residual = self.conv2(residual)
        return self.skip(x) + residual


class MultiscaleEnergyHead(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        conditioning_dim: int,
        *,
        activation_type: str = "silu",
        softplus_beta: float = 1.0,
    ):
        super().__init__()
        self.feature_projection = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        self.conditioning_projection = nn.Linear(conditioning_dim, hidden_channels, bias=False)
        self.output_projection = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.activation = _make_activation(activation_type, softplus_beta=softplus_beta)

    def forward(self, features: torch.Tensor, conditioning: torch.Tensor):
        x = self.feature_projection(features)
        x = x + self.conditioning_projection(conditioning).unsqueeze(-1).unsqueeze(-1)
        x = self.activation(x)
        x = self.output_projection(x)
        return x.mean(dim=(-1, -2))


class GlobalEnergyHead(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_dim: int,
        conditioning_dim: int,
        *,
        activation_type: str = "silu",
        softplus_beta: float = 1.0,
    ):
        super().__init__()
        self.feature_projection = nn.Linear(channels, hidden_dim)
        self.conditioning_projection = nn.Linear(conditioning_dim, hidden_dim, bias=False)
        self.output_projection = nn.Linear(hidden_dim, 1)
        self.activation = _make_activation(activation_type, softplus_beta=softplus_beta)

    def forward(self, features: torch.Tensor, conditioning: torch.Tensor):
        pooled_features = features.mean(dim=(-1, -2))
        x = self.feature_projection(pooled_features) + self.conditioning_projection(conditioning)
        x = self.activation(x)
        return self.output_projection(x)


class TimeConditionedResNet(nn.Module):
    def __init__(
        self,
        image_size: int = 28,
        input_channels: int = 1,
        conv_channels: list[int] | tuple[int, ...] = (32, 64, 128),
        blocks_per_level: int | list[int] | tuple[int, ...] = 2,
        head_hidden_dims: list[int] | tuple[int, ...] | None = None,
        global_head_hidden_dim: int = 128,
        output_dim: int = 1,
        time_embedding_dim: int = 64,
        conditioning_dim: int = 128,
        norm_groups: int = 8,
        activation_type: str = "silu",
        softplus_beta: float = 1.0,
        final_activation: str = "identity",
        energy_weight_init: float = 0.25,
    ):
        super().__init__()
        if output_dim != 1:
            raise ValueError("TimeConditionedResNet is a scalar potential and requires output_dim=1")
        if len(conv_channels) == 0:
            raise ValueError("conv_channels must contain at least one entry")
        if isinstance(blocks_per_level, int):
            blocks_per_level = [blocks_per_level] * len(conv_channels)
        if len(blocks_per_level) != len(conv_channels):
            raise ValueError("blocks_per_level must be an int or have the same length as conv_channels")
        if head_hidden_dims is None:
            head_hidden_dims = [min(channels, 64) for channels in conv_channels]
        if len(head_hidden_dims) != len(conv_channels):
            raise ValueError("head_hidden_dims must have the same length as conv_channels")

        self.image_size = image_size
        self.input_channels = input_channels
        self.final_activation = final_activation

        coordinate_values = torch.linspace(-1.0, 1.0, image_size)
        grid_y, grid_x = torch.meshgrid(coordinate_values, coordinate_values, indexing="ij")
        coordinate_channels = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)
        self.register_buffer("coordinate_channels", coordinate_channels, persistent=False)

        self.time_embedding = FourierScalarEmbedding(time_embedding_dim)
        self.conditioning_net = nn.Sequential(
            nn.Linear(time_embedding_dim, conditioning_dim),
            _make_activation(activation_type, softplus_beta=softplus_beta),
            nn.Linear(conditioning_dim, conditioning_dim),
            _make_activation(activation_type, softplus_beta=softplus_beta),
        )

        self.stem = nn.Conv2d(input_channels + 2, conv_channels[0], kernel_size=3, padding=1)
        self.level_blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        ConditionedResidualBlock(
                            channels,
                            conditioning_dim,
                            norm_groups=norm_groups,
                            activation_type=activation_type,
                            softplus_beta=softplus_beta,
                        )
                        for _ in range(num_blocks)
                    ]
                )
                for channels, num_blocks in zip(conv_channels, blocks_per_level)
            ]
        )
        self.downsample_blocks = nn.ModuleList(
            [
                ConditionedDownsampleBlock(
                    in_channels,
                    out_channels,
                    conditioning_dim,
                    norm_groups=norm_groups,
                    activation_type=activation_type,
                    softplus_beta=softplus_beta,
                )
                for in_channels, out_channels in zip(conv_channels, conv_channels[1:])
            ]
        )
        self.energy_heads = nn.ModuleList(
            [
                MultiscaleEnergyHead(
                    channels,
                    hidden_channels,
                    conditioning_dim,
                    activation_type=activation_type,
                    softplus_beta=softplus_beta,
                )
                for channels, hidden_channels in zip(conv_channels, head_hidden_dims)
            ]
        )
        self.global_energy_head = GlobalEnergyHead(
            conv_channels[-1],
            global_head_hidden_dim,
            conditioning_dim,
            activation_type=activation_type,
            softplus_beta=softplus_beta,
        )
        self.energy_weights = nn.Parameter(torch.full((len(conv_channels) + 1,), float(energy_weight_init)))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_and_h_normalized: torch.Tensor):
        x_flat = x_and_h_normalized[:, :-1]
        h_normalized = x_and_h_normalized[:, -1:].view(-1, 1)

        x = x_flat.view(-1, self.input_channels, self.image_size, self.image_size)
        coordinates = self.coordinate_channels.to(device=x.device, dtype=x.dtype).expand(x.shape[0], -1, -1, -1)
        x = torch.cat([x, coordinates], dim=1)

        conditioning = self.conditioning_net(self.time_embedding(h_normalized))
        x = self.stem(x)

        energy_terms = []
        for level_idx, blocks in enumerate(self.level_blocks):
            for block in blocks:
                x = block(x, conditioning)
            energy_terms.append(self.energy_heads[level_idx](x, conditioning))
            if level_idx < len(self.downsample_blocks):
                x = self.downsample_blocks[level_idx](x, conditioning)

        energy_terms.append(self.global_energy_head(x, conditioning))
        x = torch.stack(energy_terms, dim=0)
        x = (x * self.energy_weights.view(-1, 1, 1)).sum(dim=0)

        if self.final_activation == "sigmoid":
            return self.sigmoid(x)
        if self.final_activation == "identity":
            return x
        raise ValueError(f"Unsupported final_activation: {self.final_activation}")


def build_time_conditioned_image_model(
    model_cfg: dict,
    *,
    image_size: int,
    input_channels: int,
):
    architecture = str(model_cfg.get("architecture", "cnn")).lower()
    output_dim = int(model_cfg.get("output_dim", 1))
    time_embedding_dim = int(model_cfg.get("time_embedding_dim", 32))
    final_activation = str(model_cfg.get("final_activation", "identity"))

    if architecture == "cnn":
        return TimeConditionedCNN(
            image_size=image_size,
            input_channels=input_channels,
            conv_channels=list(model_cfg["conv_channels"]),
            fc_hidden_dim=int(model_cfg["fc_hidden_dim"]),
            output_dim=output_dim,
            time_embedding_dim=time_embedding_dim,
            use_pooling=bool(model_cfg.get("use_pooling", True)),
            pooling_type=str(model_cfg.get("pooling_type", "max")),
            activation_type=str(model_cfg.get("activation_type", "relu")),
            softplus_beta=float(model_cfg.get("softplus_beta", 1.0)),
            final_activation=final_activation,
        )

    if architecture == "resnet":
        conv_channels = list(model_cfg["conv_channels"])
        return TimeConditionedResNet(
            image_size=image_size,
            input_channels=input_channels,
            conv_channels=conv_channels,
            blocks_per_level=model_cfg.get("blocks_per_level", 2),
            head_hidden_dims=list(
                model_cfg.get("head_hidden_dims", [min(channels, 64) for channels in conv_channels])
            ),
            global_head_hidden_dim=int(model_cfg.get("global_head_hidden_dim", 128)),
            output_dim=output_dim,
            time_embedding_dim=time_embedding_dim,
            conditioning_dim=int(model_cfg.get("conditioning_dim", 128)),
            norm_groups=int(model_cfg.get("norm_groups", 8)),
            activation_type=str(model_cfg.get("activation_type", "silu")),
            softplus_beta=float(model_cfg.get("softplus_beta", 1.0)),
            final_activation=final_activation,
            energy_weight_init=float(model_cfg.get("energy_weight_init", 0.25)),
        )

    raise ValueError(f"Unsupported model architecture: {architecture}. Use 'cnn' or 'resnet'.")
