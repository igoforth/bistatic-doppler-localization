# bistatic-doppler-localization

Target localization from bistatic Doppler shift measurements using neural networks.

Given Doppler shift observations from 4 transmitters over a sequence of timesteps, predict where a moving target is located. The task is the inverse of a real tracking radar: Doppler shifts encode the target's position and velocity through nonlinear geometric relationships, and we train a model to recover position.

## Results

Variable-velocity targets on a 28×28 position grid, 100k training samples, 1000-sample held-out validation:

| Configuration | Exact | Within 1 px | Within 2 px | Mean err |
|---------------|-------|-------------|-------------|----------|
| Static single-shot MLP | 35% | 68% | 79% | 2.9 px |
| Time-series GRU (fixed velocity) | 59% | 87% | 94% | — |
| **Time-series Physics Transformer** | **60%** | **85%** | **94%** | **0.65 px** |

The Physics Transformer tokenizes the input as a grid of `(timestep, transmitter)` pairs rather than flattening each timestep, letting attention learn triangulation (cross-transmitter) and velocity inference (cross-timestep) as independent relationships.

See the [blog post](https://ian.goforth.systems/blog/bistatic-doppler-localization) for the full story: failure modes, ablations, and what actually matters for this kind of inverse problem.

## Quickstart

```bash
uv sync
uv run jupyter lab notebooks/inference.ipynb
```

Requires a GPU with PyTorch support. ROCm indexes are pre-configured in `pyproject.toml` for AMD; edit to use `+cu128` or equivalent for NVIDIA.

## Repo layout

```
src/bdl/
├── datasets/
│   ├── interface.py           # abstract ProblemDataset + DataLoader adapter
│   ├── doppler.py             # static single-shot Doppler dataset
│   └── doppler_timeseries.py  # time-series variant with linear motion
├── loss.py                    # custom_doppler_loss (exploration only)
├── inference.py               # visualization and accuracy metrics
└── constants.py

notebooks/
└── inference.ipynb            # static baseline training + visualization
```

The time-series Physics Transformer model lives in the analysis notebook and the blog post rather than the package, since the exploration wasn't meant as a library API.

## Authors

- Ian Goforth ([ian.goforth.systems](https://ian.goforth.systems))
- PaRa (bistatic geometry and data encoding design)

## License

MIT.
