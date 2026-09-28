# NKE research experiment, implementation slice 1

This repository implements the authoritative event replay state, deterministic synthetic trajectories with exact oracle labels, and a **trainable full-snapshot neural Choice model**. It is research code, not a trained NKE release. The optional model needs PyTorch, Transformers, and an accessible ModernBERT checkpoint.

## Run without a GPU or model download

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python -m nke_experiment.data --scenarios 100 --output data/generated
```

Each scenario stays in one split across all event steps. The manifest marks the labels synthetic. The reference oracle deliberately makes these toy tasks solvable with explicit code, so these numbers cannot establish enterprise intelligence.

## Train Variant A when a suitable device and model access are available

```bash
python -m pip install -e '.[model,dev]'
python -m nke_experiment.data --scenarios 1000 --output data/generated
python -m nke_experiment.train --data data/generated --output checkpoints/variant-a --limit 500 --epochs 2
python -m nke_experiment.evaluate --checkpoint checkpoints/variant-a --data data/generated --split test
```

The trainer fine-tunes only NKE projection, attention, and Choice scoring heads over a frozen licensed pretrained encoder. A saved `nke_heads.safetensors` is an actual trained head checkpoint if and only if training completed. The backbone is a separate required dependency and must be pinned to its revision for a reproducible release. Training accuracy on these synthetic tasks is not evidence of transfer. The CLI prints only validation accuracy; calibration and untouched final-test evaluation must be implemented before a model release.

The next implementation slice adds cached record embeddings (Variant B) and a learned memory updater (Variant C) under the [architecture spec](NKE_Architecture_and_Experiment_Spec_v0.1.md). Choice-only training is deliberate; Binary and ordered Score need their own supervised examples and validation.

## Current limitations

- No trained NKE weights are included.
- The synthetic state is text-serialized for the encoder. Typed numeric handling is part of the research plan.
- No service API, calibration, recurrent memory, or public benchmark has been built.
- Model downloading and training require additional environment resources. No Jev comparison exists.
