# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 11:24:40 2023

@author: PaRa
@author: igoforth
"""

import functools
import operator
from abc import ABC, abstractmethod
from typing import Any, Literal

import torch
import torch.utils.data as data


class ProblemDatasetInterface(ABC):
    @property
    @abstractmethod
    def input_shape(self) -> tuple[int, ...]:
        """
        The shape of the input tensor passed into the model in an input, output pair
        """
        pass

    @property
    @abstractmethod
    def output_shape(self) -> tuple[int, ...]:
        """
        The shape of the output tensor the model produces in an input, output pair
        """
        pass

    @property
    def input_size(self) -> int:
        """
        The input size of the tensor passed into the model in an input, output pair
        """
        return functools.reduce(operator.mul, self.input_shape, 1)

    @property
    def output_size(self) -> int:
        """
        The output size of the tensor the model produces in an input, output pair
        """
        return functools.reduce(operator.mul, self.output_shape, 1)

    @abstractmethod
    def transform_tensor(
        self, tensor: torch.Tensor, source_dim: int, target_dim: int
    ) -> torch.Tensor:
        """
        Transform input tensor between 1D, 2D, and 3D representations while preserving spatial information.

        Args:
            tensor (torch.Tensor): Input tensor of shape [batch_size, *input_shape].
            source_dim (int): The source dimensionality of the input tensor (1, 2, or 3).
            target_dim (int): The target dimensionality (1, 2, or 3).

        Returns:
            torch.Tensor: Transformed tensor with the target dimensionality while preserving spatial information.

        This method should implement transformations between all combinations of 1D, 2D, and 3D representations.
        It should strive to preserve spatial information and, where possible, make the transformations reversible.
        The specific implementation may vary depending on the nature of the data and the requirements of the problem.
        """
        pass

    @abstractmethod
    def create_dataset(
        self, num_samples: int, data_type: Literal["train", "validate"]
    ) -> None:
        """
        Create a dataset of input output pairs and hold it in memory on device.
        Args:
            num_samples (int): Number of samples to generate.
            data_type (str): Type of the data (e.g., 'train' or 'validate').
        """
        pass

    @abstractmethod
    def generate_data(self, num_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate data for the specified number of samples and type.
        Args:
            num_samples (int): Number of samples to generate.
        """
        pass

    @abstractmethod
    def get_data(
        self, index: int, data_type: Literal["train", "validate"]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get an input-target pair from the dataset by index.
        Args:
            index (int): Index of the data pair.
            data_type (str): Type of the data (e.g., 'train' or 'validate').
        Returns:
            tuple: A tuple (input, target) for the given index.
        """
        pass

    @abstractmethod
    def get_data_range(
        self, indices: list[int], data_type: Literal["train", "validate"]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get a range of input-target pairs from the dataset, given a list of indices
        Args:
            indices (list[int]): List of indices to concatenate the data from.
            data_type (str): Type of the data (e.g., 'train' or 'validate').
        Returns:
            tuple: A list of tuples (input, target) for the given index range.
        """
        pass

    @abstractmethod
    def get_len(self, data_type: Literal["train", "validate"]) -> int:
        """
        Get the number of data pairs available in the dataset.
        Args:
            data_type (str): Type of the data (e.g., 'train' or 'validate').
        Returns:
            int: Total number of data pairs.
        """
        pass

    @abstractmethod
    def get_score(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> tuple[int, int]:
        """
        Get the number of correct predictions and the total number of samples.

        Args:
            outputs (Tensor): The model predictions for the input tensors. Shape [batch_size, predictions].
            targets (Tensor): The correct class indices. Shape [batch_size].

        Returns:
            tuple[int, int]: A tuple where the first element is the number of correct predictions
                             and the second element is the total number of samples.
        """
        pass

    @abstractmethod
    def preprocess_data(
        self,
        data_type: Literal["train", "validate"],
        normalize: bool = False,
        add_noise: bool = False,
    ) -> None:
        """
        Apply preprocessing steps to the data such as normalization and noise addition.
        Args:
            normalize (bool): Whether to normalize the data.
            add_noise (bool): Whether to add noise to the data.
        """
        pass


class DatasetAdapter(data.Dataset[Any]):
    def __init__(
        self,
        problem: ProblemDatasetInterface,
        data_len: int,
        data_type: Literal["train", "validate"] = "train",
    ):
        self.problem = problem
        self.data_len = data_len
        self.data_type: Literal["train", "validate"] = data_type

        if self.data_len > self.problem.get_len(self.data_type):
            self.problem.create_dataset(self.data_len, data_type)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.problem.get_data(index, self.data_type)

    def __getitems__(self, indices: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        return self.problem.get_data_range(indices, self.data_type)

    def __len__(self):
        return self.data_len
