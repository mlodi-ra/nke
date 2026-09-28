# NKE research experiment, implementation slice 1

This repository implements the authoritative event replay state, deterministic synthetic trajectories with exact oracle labels, and a **trainable full-snapshot neural Choice model**. It is research code, not a trained NKE release. The optional model needs PyTorch, Transformers, and an accessible ModernBERT checkpoint.

## Run without a GPU or model download

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python -m nke_experiment.data --scenarios 100 --output data/generated
python -m nke_experiment.diagnose --data data/generated --split test
```

Each scenario stays in one split across all event steps. The manifest marks the labels synthetic. The reference oracle deliberately makes these toy tasks solvable with explicit code, so these numbers cannot establish enterprise intelligence.

## Train Variant A when a suitable device and model access are available

```bash
python -m pip install -e '.[model,dev]'
python -m nke_experiment.data --scenarios 1000 --output data/generated
python -m nke_experiment.train --data data/generated --output checkpoints/variant-a --limit 500 --epochs 2
python -m nke_experiment.diagnose --checkpoint checkpoints/variant-a --data data/generated --split validation
```

The trainer selects complete scenarios across all task families, then fine-tunes only NKE projection, attention, and Choice scoring heads over a frozen licensed pretrained encoder. Earlier runs with `--limit` selected the first rows from a file ordered by family. At the documented 500-row limit, priority cases were excluded and validation coverage was biased. Re-run training before interpreting those results. A saved `nke_heads.safetensors` is an actual trained head checkpoint if and only if training completed. The backbone is a separate required dependency and must be pinned to its revision for a reproducible release. Training accuracy on these synthetic tasks is not evidence of transfer. Validation accuracy is reported by family. Compare a saved checkpoint on the untouched test split with `python -m nke_experiment.diagnose --data data/generated --checkpoint checkpoints/variant-a --split test`. The CLI reports majority baselines learned from the training split, model accuracy by family and step, and sample errors. These diagnostics do not establish transfer or calibration.

New checkpoints use an experimental typed numeric channel and attach a matching record to each candidate. These are generic field encodings, not an implementation of the task oracle. Access and priority have exhaustive choices; resource assignment permits `none` when no worker qualifies. Older checkpoints retain their original architecture when loaded. The original 100-scenario test split has already informed development and must not serve as the final test for this revision. The [validation record](experiments/2026-09-27-typed-input-validation.md) shows an improvement on resource assignment but not access or priority. After selecting an architecture that passes the validation gate, create new scenario IDs and run one final evaluation:

```bash
python -m nke_experiment.holdout --output data/final-holdout --start-id 10000 --scenarios 200
python -m nke_experiment.diagnose --data data/generated --held-out-data data/final-holdout --checkpoint checkpoints/variant-a --split test
```

These remain synthetic tests with a narrow oracle. Do not tune on the final holdout results.

The next implementation slice adds cached record embeddings (Variant B) and a learned memory updater (Variant C) under the [architecture spec](NKE_Architecture_and_Experiment_Spec_v0.1.md). Choice-only training is deliberate; Binary and ordered Score need their own supervised examples and validation.

## Current limitations

- No trained NKE weights are included.
- The synthetic state is text-serialized for the encoder. Typed numeric handling is part of the research plan.
- No service API, calibration, recurrent memory, or public benchmark has been built.
- Model downloading and training require additional environment resources. No Jev comparison exists.
