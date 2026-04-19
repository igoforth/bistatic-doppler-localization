---
title: "Target Localization from Bistatic Doppler Shift Vectors"
description: "Synthetic radar data generation, a custom multi-term loss function for sparse position estimation, and what happens when your model outputs aren't bounded."
pubDate: "Apr 3 2026"
---

## Introduction

A bistatic radar system has transmitters and receivers at different locations. Each transmitter illuminates a target, and the receiver measures the Doppler shift — the frequency change caused by the target's motion relative to both the transmitter and receiver. With multiple transmitters, you get multiple Doppler measurements from different geometric perspectives. The question is whether a neural network can learn the inverse mapping: given a set of Doppler shift vectors, predict where the target is.

PaRa designed the bistatic geometry and the data encoding scheme. I implemented the dataset pipeline, loss function, training infrastructure, and inference visualization. This post covers the data generation, the loss function design, and a training failure mode that took longer to diagnose than it should have.

This dataset was originally built for a genetic algorithm-driven neural architecture search system that evolves MLP, Transformer, and KAN architectures. The notebook here trains a simple MLP baseline to validate the dataset and loss function independently of the NAS framework.

## The Bistatic Geometry

Four transmitters are placed around a 28,000m square region:

```python
self.transmitters = torch.tensor([
    [-36000, -36000, 140e6],  # top left
    [ 64000, -36000, 140e6],  # top right
    [-36000,  64000, 140e6],  # bottom left
    [ 64000,  64000, 140e6],  # bottom right
])
```

Each transmitter broadcasts at 140 MHz. A target at position `(x, y)` moving with velocity `(xdot, ydot)` produces a Doppler shift relative to each transmitter:

```python
n1 = xdot * x + ydot * y
n2 = xdot * (x - xn) + ydot * (y - yn)
d1 = torch.sqrt(x**2 + y**2)
d2 = torch.sqrt((x - xn)**2 + (y - yn)**2)
m = -(F / C) * (n1/d1 + n2/d2)
```

The shift depends on the target's position, velocity, and the transmitter location. Each shift value is quantized into a 1000-element vector (a histogram bin centered at the shift frequency). The result is a `(4, 1000)` input tensor per sample — four Doppler spectra, one per transmitter.

The target output is a 28x28 image with a single pixel set to 1.0 at the target's position. Flattened to 784 elements.

## Data Generation

All data is generated synthetically on GPU. No files, no I/O bottleneck:

```python
problem = DopplerDataset(device)
problem.create_dataset(10000, "train", use_cache=False)
problem.preprocess_data("train", normalize=True, make_contiguous=True)
```

Positions are sampled uniformly over the region, velocities between 50-150 m/s. The Doppler computation is fully vectorized — generating 10,000 samples takes under a second on a Radeon RX 6700 XT.

Normalization is per-transmitter: mean and standard deviation computed across samples and frequency bins, then applied in-place. This matters because each transmitter's geometric perspective produces a different Doppler shift distribution.

<!-- [screenshot: sample input visualization showing 4 Doppler spectra] -->

## The Loss Function

MSE alone doesn't work for this problem. The target is a 28x28 image that's 99.87% zeros with a single nonzero pixel. MSE treats every pixel equally, so a model that outputs all zeros gets a loss of ~0.0013 — already very low. There's no gradient signal to find the peak.

`custom_doppler_loss` combines five terms:

```python
total_loss = (
    mse_loss
    + peak_weight * peak_loss           # 15x
    + shift_weight * shift_loss         # 10x
    + spatial_weight * spatial_loss     # 8x
    + background_weight * background_loss  # 5x
)
```

**Peak intensity loss.** MSE between the maximum predicted value and the maximum target value. Forces the model to produce a strong peak, not a uniform low-energy field.

**Shift loss.** L1 distance between the argmax positions of prediction and target. The model gets penalized for putting the peak in the wrong place, even if the peak intensity is correct.

**Spatial loss.** MSE between the soft center-of-mass of prediction and target, computed via softmax-weighted coordinate averaging. This provides smooth gradients even when the argmax is far from the target — unlike shift loss, which has zero gradient everywhere except at discrete transitions.

**Background suppression.** Mean of predicted values where the target is zero. Penalizes the model for activating background pixels.

The weights (15, 10, 8, 5) were tuned empirically. The spatial and shift terms do most of the work during early training. Peak and background refine the output once the model is roughly locating the target.

## The Sigmoid Problem

The first version of the baseline MLP had no output activation:

```python
model = nn.Sequential(
    nn.Linear(4000, 512), nn.BatchNorm1d(512), nn.ReLU(),
    nn.Linear(512, 512),  nn.BatchNorm1d(512), nn.ReLU(),
    nn.Linear(512, 256),  nn.BatchNorm1d(256), nn.ReLU(),
    nn.Linear(256, 784),
)
```

It trained to a loss of ~135 and produced outputs centered around -2.5 with no visible structure. 0% accuracy.

The issue: unbounded outputs interact badly with the loss terms. The background suppression loss penalizes `mean(pred[target == 0])`, but without bounding, the model can minimize this by pushing everything negative. The peak loss compares `max(pred)` to `max(target)` = 1.0, but a model outputting values in [-3, 1] gets decent peak loss while having terrible spatial structure. The spatial loss uses softmax over the flattened output, and softmax over unbounded negative values produces a nearly uniform distribution — no gradient signal for localization.

Adding `nn.Sigmoid()` to the output fixes everything:

```python
    nn.Linear(256, 784),
    nn.Sigmoid(),
```

Outputs are now in [0, 1]. Background suppression has clear gradients (push toward 0). Peak loss has a meaningful target (push max toward 1). Softmax over sigmoid outputs preserves the spatial structure.

Same model, same loss, same hyperparameters. 99.7% accuracy.

<!-- [screenshot: training loss curve showing convergence] -->

## Results

The MLP baseline trains in ~2 minutes on a Radeon RX 6700 XT (ROCm 6.4) with 10,000 training samples and 30 epochs.

<!-- [screenshot: predicted vs actual Doppler maps side-by-side, 3-4 samples] -->

<!-- [screenshot: difference maps showing prediction error] -->

The model consistently places the peak within 1 pixel of the true position. The main failure mode is slight energy dispersion around the peak — the predicted map has a soft halo where the target map has a single hard pixel.

<!-- [screenshot: MSE loss heatmap and sample count heatmap from analysis] -->

The MSE heatmap shows error is roughly uniform across the target space, with no systematic bias toward edges or corners. The sample count heatmap confirms the uniform position sampling.

## Architecture

The project is a Python package (`bdl`) with four modules:

```
src/bdl/
├── datasets/
│   ├── interface.py    # abstract dataset interface + DataLoader adapter
│   └── doppler.py      # bistatic Doppler data generation
├── loss.py             # custom multi-term Doppler loss
├── inference.py        # visualization and accuracy metrics
└── constants.py
```

The `DopplerDataset` pre-allocates GPU tensors at import time and generates data in-place. The `DatasetAdapter` implements PyTorch's `Dataset` interface with batch-fetched `__getitems__` for efficient GPU-resident data loading — no CPU-GPU transfer during training.

The code is at [github.com/igoforth/bistatic-doppler-localization](https://github.com/igoforth/bistatic-doppler-localization).
