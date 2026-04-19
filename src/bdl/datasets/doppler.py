# -*- coding: utf-8 -*-
"""
@author: PaRa
@author: igoforth
"""

import fcntl
import os
import pickle
import shutil
import tempfile
from typing import Literal, Self

import torch

from bdl.constants import MAX_TRAIN_SAMPLES, MAX_VALIDATION_SAMPLES

from .interface import ProblemDatasetInterface


class DopplerDataset(ProblemDatasetInterface):
    _instance: Self | None = None
    MAX_TRAIN_SAMPLES = MAX_TRAIN_SAMPLES
    MAX_VALIDATION_SAMPLES = MAX_VALIDATION_SAMPLES

    def __new__(cls, *args, **kwargs):  # type: ignore
        if not cls._instance:
            cls._instance = super(DopplerDataset, cls).__new__(cls)
        return cls._instance

    def __init__(self, device: torch.device, dtype: torch.dtype = torch.float32):
        if not hasattr(self, "initialized"):
            self.device = device
            self.dtype = dtype

            self.num_transmitters: int = 4
            self.image_size: int = 28
            self.vector_size: int = 1000
            self.velocity_range = (75, 75)
            self.position_range = (0, self.image_size * 1000)

            self.C_T = torch.tensor(
                299792458,
                dtype=self.dtype,
                device=self.device,
            )
            self.F_T = torch.tensor(
                140e6,
                dtype=self.dtype,
                device=self.device,
            )
            self.M1_T = -(self.F_T / self.C_T)

            self.transmitters = torch.tensor(
                [
                    [-36000, -36000, self.F_T],  # top left
                    [64000, -36000, self.F_T],  # top right
                    [-36000, 64000, self.F_T],  # bottom left
                    [64000, 64000, self.F_T],  # bottom right
                ],
                dtype=self.dtype,
                device=self.device,
            )

            # Derive input_shape and output_shape
            self._input_shape = (self.num_transmitters, self.vector_size)
            self._output_shape = (self.image_size * self.image_size,)

            # Pre-allocate memory for inputs and targets separately
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
        if target_dim == 1:
            return tensor.view(tensor.size(0), -1)
        elif target_dim == 2:
            batch_size = tensor.size(0)
            return tensor.view(batch_size, self.num_transmitters, self.vector_size)
        elif target_dim == 3:
            batch_size = tensor.size(0)
            return tensor.view(
                batch_size, self.num_transmitters, 1, self.vector_size
            )
        raise ValueError(f"Invalid target_dim {target_dim}. Must be 1, 2, or 3.")

    def get_data(
        self, index: int, data_type: Literal["train", "validate"] = "train"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if data_type == "train":
            return self.train_data[index], self.train_targets[index]
        else:
            return self.val_data[index], self.val_targets[index]

    def get_data_range(
        self, indices: list[int], data_type: Literal["train", "validate"] = "train"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if data_type == "train":
            return self.train_data[indices], self.train_targets[indices]
        else:
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
        cache_folder = "cache/"
        cache_file = cache_folder + f"doppler_cache_{data_type}.pkl"
        lock_file = f"{cache_file}.lock"

        if use_cache:
            self._load_from_cache(cache_file, lock_file, num_samples, data_type)

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

            new_inputs = new_inputs.view(
                new_samples, self.num_transmitters, self.vector_size
            )
            new_targets = new_targets.view(new_samples, -1)

            data[current_count : current_count + new_samples] = new_inputs
            targets[current_count : current_count + new_samples] = new_targets

            if data_type == "train":
                self.train_samples = current_count + new_samples
            else:
                self.val_samples = current_count + new_samples

        if use_cache:
            self._append_to_cache(cache_file, lock_file, data_type)

    def _get_doppler_torch(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        xdot: torch.Tensor,
        ydot: torch.Tensor,
        xn: torch.Tensor,
        yn: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate the Doppler shift for given positions and velocities.
        Supports both single inputs and batched inputs.
        """
        tensors = [x, y, xdot, ydot, xn, yn]
        tensors = [
            t.unsqueeze(0).to(self.device) if t.dim() == 1 else t.to(self.device)
            for t in tensors
        ]
        x, y, xdot, ydot, xn, yn = tensors

        n1 = xdot * x + ydot * y
        n2 = xdot * (x - xn) + ydot * (y - yn)
        d1 = torch.sqrt(x**2 + y**2)
        d2 = torch.sqrt((x - xn) ** 2 + (y - yn) ** 2)

        t1 = n1 / d1
        t2 = n2 / d2

        m = self.M1_T * (t1 + t2)
        m = torch.nan_to_num(m)

        return m.squeeze()

    def generate_data(self, num_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        xs = torch.randint(
            low=self.position_range[0],
            high=self.position_range[1],
            size=(num_samples,),
            device=self.device,
        )
        ys = torch.randint(
            low=self.position_range[0],
            high=self.position_range[1],
            size=(num_samples,),
            device=self.device,
        )
        xdots = (
            torch.rand(num_samples, dtype=self.dtype, device=self.device)
            * (self.velocity_range[1] - self.velocity_range[0])
            + self.velocity_range[0]
        )
        ydots = (
            torch.rand(num_samples, dtype=self.dtype, device=self.device)
            * (self.velocity_range[1] - self.velocity_range[0])
            + self.velocity_range[0]
        )

        vec = torch.zeros(
            (num_samples, self.num_transmitters, self.vector_size),
            dtype=self.dtype,
            device=self.device,
        )
        ima = torch.zeros(
            (num_samples, self.image_size, self.image_size),
            dtype=self.dtype,
            device=self.device,
        )

        for i, (xn, yn, _freq) in enumerate(self.transmitters):
            m_values = self._get_doppler_torch(xs, ys, xdots, ydots, xn, yn)
            indices = torch.floor(m_values + (self.vector_size / 2)).long()
            valid_mask = (indices >= 0) & (indices < self.vector_size)
            indices = indices[valid_mask]
            batch_indices = torch.arange(num_samples, device=indices.device)[valid_mask]
            vec[batch_indices, i, indices] += 1.0

        scaled_xs = torch.round(
            (xs / 1000) * (ima.shape[1] - 1) / (self.image_size - 1)
        )
        scaled_ys = torch.round(
            (ys / 1000) * (ima.shape[2] - 1) / (self.image_size - 1)
        )
        scaled_xs = torch.clamp(scaled_xs, 0, ima.shape[1] - 1).long()
        scaled_ys = torch.clamp(scaled_ys, 0, ima.shape[2] - 1).long()
        ima[torch.arange(num_samples), scaled_xs, scaled_ys] = 1.0

        return vec, ima

    def _load_from_cache(
        self, cache_file: str, lock_file: str, num_samples: int, data_type: str
    ):
        with open(lock_file, "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                if os.path.exists(cache_file):
                    with open(cache_file, "rb") as f:
                        cached_data: dict[str, torch.Tensor] = pickle.load(f)
                        inputs = cached_data["inputs"][:num_samples].to(
                            self.device, non_blocking=True
                        )
                        targets = cached_data["targets"][:num_samples].to(
                            self.device, non_blocking=True
                        )
                        if data_type == "train":
                            self.train_data[:num_samples] = inputs
                            self.train_targets[:num_samples] = targets.view(
                                num_samples, -1
                            )
                            self.train_samples = num_samples
                        else:
                            self.val_data[:num_samples] = inputs
                            self.val_targets[:num_samples] = targets.view(
                                num_samples, -1
                            )
                            self.val_samples = num_samples
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _append_to_cache(self, cache_file: str, lock_file: str, data_type: str):
        with open(lock_file, "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                with tempfile.NamedTemporaryFile(mode="wb", delete=False) as temp_file:
                    if data_type == "train":
                        inputs = self.train_data[: self.train_samples]
                        targets = self.train_targets[: self.train_samples]
                    else:
                        inputs = self.val_data[: self.val_samples]
                        targets = self.val_targets[: self.val_samples]

                    pickle.dump(
                        {
                            "inputs": inputs.cpu(),
                            "targets": targets.cpu(),
                        },
                        temp_file,
                    )
                    shutil.move(temp_file.name, cache_file)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def preprocess_data(
        self,
        data_type: str,
        normalize: bool = False,
        add_noise: bool = False,
        make_contiguous: bool = True,
    ) -> None:
        if add_noise:
            self._generate_spectral_noise(data_type)

        if normalize:
            self._normalize_inputs(data_type)

        if make_contiguous:
            self._make_contiguous(data_type)

    def _generate_spectral_noise(self, data_type: str):
        if data_type == "train":
            scale = 0.025
            data = self.train_data[: self.train_samples]
        else:
            scale = 0.1
            data = self.val_data[: self.val_samples]

        noise = torch.randn(
            data.shape, dtype=torch.cfloat, device=self.device
        )
        noise_fft = torch.fft.fft(noise, dim=-1)
        noise_fft_scaled = noise_fft * scale
        time_domain_noise = torch.fft.ifft(noise_fft_scaled, dim=-1)
        data += time_domain_noise.real

    def _normalize_inputs(self, data_type: str):
        if data_type == "train":
            data = self.train_data[: self.train_samples]
        else:
            data = self.val_data[: self.val_samples]

        mean = data.mean(dim=(0, 2), keepdim=True)
        std = data.std(dim=(0, 2), keepdim=True)
        data[:] = (data - mean) / std

    def _make_contiguous(self, data_type: str):
        if data_type == "train":
            if not self.train_data.is_contiguous():
                self.train_data = self.train_data.contiguous()
        else:
            if not self.val_data.is_contiguous():
                self.val_data = self.val_data.contiguous()
