# -*- coding: utf-8 -*-
"""
Time-series bistatic Doppler localization dataset.

Target moves linearly with constant velocity for T timesteps. At each timestep,
bistatic Doppler is computed from all 4 transmitters. The model gets a sequence
of Doppler observations and predicts the target's initial position.

@author: PaRa
@author: igoforth
"""

from typing import Literal, Self

import torch

from bdl.constants import DEVICE

from .interface import ProblemDatasetInterface


class DopplerTimeSeriesDataset(ProblemDatasetInterface):
    _instance: Self | None = None
    MAX_TRAIN_SAMPLES = 10000
    MAX_VALIDATION_SAMPLES = 2000

    def __new__(cls, *args, **kwargs):  # type: ignore
        if not cls._instance:
            cls._instance = super(DopplerTimeSeriesDataset, cls).__new__(cls)
        return cls._instance

    def __init__(
        self,
        device: torch.device = DEVICE,
        dtype: torch.dtype = torch.float32,
        num_timesteps: int = 5,
        dt: float = 10.0,
    ):
        if not hasattr(self, "initialized"):
            self.device = device
            self.dtype = dtype

            self.num_transmitters: int = 4
            self.image_size: int = 28
            self.vector_size: int = 1000
            self.num_timesteps: int = num_timesteps
            self.dt: float = dt

            self.speed_range = (50.0, 150.0)  # magnitude, direction randomized
            self.position_range = (0, self.image_size * 1000)

            self.C_T = torch.tensor(299792458, dtype=self.dtype, device=self.device)

            # Per-transmitter carrier frequencies. Real bistatic radar systems
            # use FDMA (one frequency per transmitter) so the receiver can
            # separate reflections. Identical frequencies also preserves
            # geometric symmetries that produce mirror-image Doppler patterns.
            self.tx_frequencies = torch.tensor(
                [140e6, 145e6, 150e6, 155e6],
                dtype=self.dtype,
                device=self.device,
            )

            self.transmitters = torch.tensor(
                [
                    [-36000, -36000],
                    [64000, -36000],
                    [-36000, 64000],
                    [64000, 64000],
                ],
                dtype=self.dtype,
                device=self.device,
            )

            self._input_shape = (
                self.num_timesteps,
                self.num_transmitters,
                self.vector_size,
            )
            self._output_shape = (self.image_size * self.image_size,)

            self.train_data = torch.empty(
                (self.MAX_TRAIN_SAMPLES, *self.input_shape),
                dtype=self.dtype,
                device=self.device,
            )
            self.train_targets = torch.empty(
                (self.MAX_TRAIN_SAMPLES, *self.output_shape),
                dtype=self.dtype,
                device=self.device,
            )
            self.val_data = torch.empty(
                (self.MAX_VALIDATION_SAMPLES, *self.input_shape),
                dtype=self.dtype,
                device=self.device,
            )
            self.val_targets = torch.empty(
                (self.MAX_VALIDATION_SAMPLES, *self.output_shape),
                dtype=self.dtype,
                device=self.device,
            )

            self.train_samples: int = 0
            self.val_samples: int = 0

            self.initialized = True

    @property
    def input_shape(self) -> tuple[int, ...]:
        return self._input_shape

    @property
    def output_shape(self) -> tuple[int, ...]:
        return self._output_shape

    def transform_tensor(
        self, tensor: torch.Tensor, source_dim: int, target_dim: int
    ) -> torch.Tensor:
        if source_dim == target_dim:
            return tensor
        batch_size = tensor.size(0)
        if target_dim == 1:
            return tensor.view(batch_size, -1)
        elif target_dim == 2:
            return tensor.view(
                batch_size, self.num_timesteps, self.num_transmitters * self.vector_size
            )
        elif target_dim == 3:
            return tensor.view(
                batch_size, self.num_timesteps, self.num_transmitters, self.vector_size
            )
        raise ValueError(f"Invalid target_dim {target_dim}. Must be 1, 2, or 3.")

    def get_data(
        self, index: int, data_type: Literal["train", "validate"] = "train"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if data_type == "train":
            return self.train_data[index], self.train_targets[index]
        return self.val_data[index], self.val_targets[index]

    def get_data_range(
        self, indices: list[int], data_type: Literal["train", "validate"] = "train"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if data_type == "train":
            return self.train_data[indices], self.train_targets[indices]
        return self.val_data[indices], self.val_targets[indices]

    def get_len(self, data_type: Literal["train", "validate"]) -> int:
        return self.train_samples if data_type == "train" else self.val_samples

    def get_score(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> tuple[int, int]:
        _, predicted = torch.max(outputs, 1)
        correct = int((predicted == targets).sum().item())
        return correct, len(targets)

    def create_dataset(
        self,
        num_samples: int,
        data_type: Literal["train", "validate"] = "train",
        use_cache: bool = False,
    ) -> None:
        if data_type == "train":
            data = self.train_data
            targets = self.train_targets
            current_count = self.train_samples
        else:
            data = self.val_data
            targets = self.val_targets
            current_count = self.val_samples

        if current_count < num_samples:
            new_samples = num_samples - current_count
            new_inputs, new_targets = self.generate_data(new_samples)

            data[current_count : current_count + new_samples] = new_inputs
            targets[current_count : current_count + new_samples] = new_targets.view(
                new_samples, -1
            )

            if data_type == "train":
                self.train_samples = current_count + new_samples
            else:
                self.val_samples = current_count + new_samples

    def preprocess_data(
        self,
        data_type: str,
        normalize: bool = False,
        add_noise: bool = False,
        make_contiguous: bool = True,
    ) -> None:
        if normalize:
            self._normalize_inputs(data_type)
        if make_contiguous:
            self._make_contiguous(data_type)

    def _doppler_torch(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        xdot: torch.Tensor,
        ydot: torch.Tensor,
        xn: torch.Tensor,
        yn: torch.Tensor,
        F: torch.Tensor,
    ) -> torch.Tensor:
        """Bistatic Doppler shift for a transmitter at (xn, yn) with carrier F.

        All input tensors are (N,); F is a scalar.
        """
        n1 = xdot * x + ydot * y
        n2 = xdot * (x - xn) + ydot * (y - yn)
        d1 = torch.sqrt(x**2 + y**2)
        d2 = torch.sqrt((x - xn) ** 2 + (y - yn) ** 2)
        m = -(F / self.C_T) * (n1 / d1 + n2 / d2)
        return torch.nan_to_num(m).squeeze()

    def generate_data(
        self, num_samples: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Initial positions
        xs = torch.randint(
            low=self.position_range[0],
            high=self.position_range[1],
            size=(num_samples,),
            device=self.device,
            dtype=self.dtype,
        ).to(self.dtype)
        ys = torch.randint(
            low=self.position_range[0],
            high=self.position_range[1],
            size=(num_samples,),
            device=self.device,
            dtype=self.dtype,
        ).to(self.dtype)

        # Velocities: positive components in [speed_range], matching original
        # DopplerDataset convention. Restricting to quadrant 1 reduces geometric
        # ambiguity (mirrored trajectories can produce similar Doppler patterns).
        vmin, vmax = self.speed_range
        vx = (
            torch.rand(num_samples, dtype=self.dtype, device=self.device)
            * (vmax - vmin)
            + vmin
        )
        vy = (
            torch.rand(num_samples, dtype=self.dtype, device=self.device)
            * (vmax - vmin)
            + vmin
        )

        # Alternative: uniform speed magnitude, uniform direction in [0, 2π).
        # More varied but introduces geometric ambiguities that hurt generalization.
        # speed = (
        #     torch.rand(num_samples, dtype=self.dtype, device=self.device)
        #     * (vmax - vmin) + vmin
        # )
        # angle = (
        #     torch.rand(num_samples, dtype=self.dtype, device=self.device)
        #     * 2 * torch.pi
        # )
        # vx = speed * torch.cos(angle)
        # vy = speed * torch.sin(angle)

        # Pre-allocate output
        vec = torch.zeros(
            (num_samples, self.num_timesteps, self.num_transmitters, self.vector_size),
            dtype=self.dtype,
            device=self.device,
        )
        batch_idx = torch.arange(num_samples, device=self.device)

        # Compute Doppler at each timestep for each transmitter
        for t in range(self.num_timesteps):
            x_t = xs + vx * (t * self.dt)
            y_t = ys + vy * (t * self.dt)
            for i, (xn, yn) in enumerate(self.transmitters):
                F = self.tx_frequencies[i]
                m_values = self._doppler_torch(x_t, y_t, vx, vy, xn, yn, F)
                indices = torch.floor(m_values + (self.vector_size / 2)).long()
                valid_mask = (indices >= 0) & (indices < self.vector_size)
                valid_batch = batch_idx[valid_mask]
                valid_indices = indices[valid_mask]
                vec[valid_batch, t, i, valid_indices] += 1.0

        # Target: position at t=0
        ima = torch.zeros(
            (num_samples, self.image_size, self.image_size),
            dtype=self.dtype,
            device=self.device,
        )
        scaled_xs = torch.round(xs / 1000).clamp(0, self.image_size - 1).long()
        scaled_ys = torch.round(ys / 1000).clamp(0, self.image_size - 1).long()
        ima[batch_idx, scaled_xs, scaled_ys] = 1.0

        return vec, ima

    def _normalize_inputs(self, data_type: str):
        if data_type == "train":
            data = self.train_data[: self.train_samples]
        else:
            data = self.val_data[: self.val_samples]

        # Normalize per-(timestep, transmitter) across samples and Doppler bins.
        # Chunked in-place to avoid a full-size temporary allocation.
        mean = data.mean(dim=(0, 3), keepdim=True)
        std = data.std(dim=(0, 3), keepdim=True)
        chunk = 1000
        for i in range(0, data.size(0), chunk):
            end = min(i + chunk, data.size(0))
            data[i:end].sub_(mean).div_(std + 1e-8)

    def _make_contiguous(self, data_type: str):
        if data_type == "train":
            if not self.train_data.is_contiguous():
                self.train_data = self.train_data.contiguous()
        else:
            if not self.val_data.is_contiguous():
                self.val_data = self.val_data.contiguous()
