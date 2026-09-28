# NKE Project Status and Roadmap

**Last updated:** 27 September 2026  
**Repository:** [mlodi-ra/nke](https://github.com/mlodi-ra/nke)  
**Status:** Active research prototype. Not ready for a public model release or enterprise use.

## 1. Objective

NKE is an experiment in answering typed decision questions from changing state. The central hypothesis is that a model can retain or reuse representations of enterprise records, update only what changed, and answer new questions without repeatedly processing the entire state.

The research must distinguish three different ideas:

1. **Full recomputation, Variant A:** Encode the current snapshot and answer the question.
2. **Cached record representations, Variant B:** Reuse encodings for unchanged records and update only changed records.
3. **Learned persistent memory, Variant C:** Maintain and update a compact learned state across an event sequence.

Variant B is an engineering optimization. Variant C is the actual architectural research claim. Variant C must outperform Variant B at comparable quality before we can claim that learned persistent memory adds value.

## 2. What has been completed

### Architecture and experiment design

- Defined versioned state records and ordered upsert/delete events.
- Defined the A, B and C comparison needed to test the core hypothesis.
- Defined evaluation gates for prediction quality, cache equivalence, learned-memory value, long histories, transfer and uncertainty.
- Chose ModernBERT-base as the initial frozen encoder, not as a permanent or proven best choice.
- Established that synthetic tasks are plumbing and architecture tests, not evidence of general enterprise intelligence.

### State and data pipeline

- Implemented an authoritative `StateStore` with strict version sequencing.
- Implemented deterministic replay of complete event histories.
- Added validation for missing versions and invalid deletions.
- Prevented snapshots from mutating the underlying state.
- Created deterministic synthetic trajectories for three task families:
  - Resource assignment based on skill, capacity and cost.
  - Access decisions based on roles and policy.
  - Incident priority based on severity and affected services.
- Kept all steps from one scenario in the same train, validation, calibration or test split.
- Added a symbolic oracle that supplies exact labels for the controlled tasks.
- Added a separate scenario-ID range for a future final holdout.

### Trainable Variant A

- Implemented a full-snapshot Choice model using a frozen ModernBERT encoder.
- Added trainable projection, pooling, cross-attention and candidate-scoring heads.
- Added checkpoint saving and loading with architecture-version compatibility.
- Added explicit semantics for whether `none` is an available answer.
- Added candidate-to-record binding so a worker candidate can directly access its current record.
- Added a schema-neutral numeric channel so numeric values are not available only as text tokens.
- Kept older checkpoints loadable under the original architecture.

### Evaluation and diagnostics

- Added accuracy, Brier score and log-loss evaluation by task family.
- Added training-split majority baselines.
- Added accuracy by trajectory step and representative error examples.
- Added deterministic, balanced selection of complete scenarios across task families.
- Added eight passing local tests covering replay, data isolation, sampling, baselines, numeric handling and final-holdout separation.
- Added continuous integration for replay, data generation and unit tests.
- Released the repository under the MIT License.

## 3. Problems found and corrected

### Biased training selection

The first `--limit 500` implementation selected the first 500 rows from a file ordered by task family. It trained on:

| Family | Rows in first run |
| --- | ---: |
| Resource | 365 |
| Access | 135 |
| Priority | 0 |

That invalidated the original cross-family interpretation. The corrected sampler keeps complete five-step scenarios together and produced this 500-row mix:

| Family | Rows after correction |
| --- | ---: |
| Resource | 165 |
| Access | 170 |
| Priority | 165 |

### Weak candidate and numeric representation

The original scorer saw candidate names such as `worker:0`, but it did not directly bind each candidate to that worker's current skill, capacity and cost. Numeric values also reached the model only through serialized text. The current revision adds both candidate-record binding and a typed numeric channel.

### Incorrect use of `none`

The original model always added `none` as a possible answer. That was valid for resource assignment, where no worker may qualify, but not for the exhaustive access and priority questions. Choice availability is now explicit in each example.

### Test-set contamination

The original test split was inspected while diagnosing model failures. It can no longer serve as an untouched final test for the revised model. A separate scenario range beginning at ID 10,000 is reserved for the final evaluation and remains unused.

## 4. Current experimental results

The latest exploratory CPU run used:

- 500 balanced training rows.
- All 140 validation rows.
- A frozen ModernBERT-base encoder.
- Trainable NKE heads and typed-input components.
- Seed 17.
- Two training epochs.

| Task family | Validation rows | Majority baseline | Model, epoch 1 | Model, epoch 2 |
| --- | ---: | ---: | ---: | ---: |
| Access | 60 | 36/60, 60.0% | 36/60, 60.0% | 36/60, 60.0% |
| Priority | 35 | 24/35, 68.6% | 24/35, 68.6% | 24/35, 68.6% |
| Resource | 45 | 15/45, 33.3% | 26/45, 57.8% | 39/45, 86.7% |

### Bottom line

The revised inputs produced a substantial validation improvement on resource assignment. Access and priority still only matched their majority baselines. The experiment therefore failed the planned gate requiring improvement over a strong simple baseline on at least two task families.

These results are too small and correlated to support statistical significance or any enterprise capability claim. The trained checkpoint remains local and has not been published as a model release.

## 5. What has not been built

- A model that passes the prediction-quality gate on at least two task families.
- Multiple-seed experiments or confidence intervals.
- Strong lexical and classical machine-learning baselines.
- A valid untouched final evaluation for the current architecture.
- Variant B cached record embeddings and cache-equivalence tests.
- Variant C learned persistent memory and sequence training.
- Long-history, deletion, correction and missed-event stress tests for learned memory.
- Calibration, expected calibration error, risk/coverage or abstention thresholds.
- Binary and ordered Score heads.
- A separately reviewed, licensed, human-labeled benchmark.
- A service API, user interface or enterprise integration.
- A valid comparison with Jev or another external system.
- A trained checkpoint suitable for public release.

## 6. Next phase

### Step 1: Strengthen the prediction experiment

Before implementing memory, improve Variant A enough to determine whether the readout can learn the controlled decisions.

- Expand the generator with more diverse roles, policies, priority thresholds, worker counts, ties, missing values, corrections, deletes and distractors.
- Separate scenario templates and entity patterns across splits, not only scenario IDs.
- Add task-family-balanced batches or losses so one family cannot dominate optimization.
- Test whether a shared scorer is appropriate or whether task-conditioned readout heads are required.
- Compare frozen-backbone training with selective fine-tuning of the last encoder layers.
- Run at least three seeds for shortlisted configurations.
- Evaluate on the validation and calibration splits only during model selection.

### Step 2: Add meaningful baselines

Compare the neural model against:

- Majority baseline.
- Lexical or nearest-neighbor baseline.
- Classical learned baseline using structured fields.
- Symbolic oracle as the ceiling for these controlled tasks.

The neural model should beat the strongest non-oracle baseline on at least two families, with uncertainty reported by independent trajectory rather than by correlated rows.

### Step 3: Freeze the architecture and run one final evaluation

Only after Steps 1 and 2 pass validation:

1. Freeze the architecture, weights-selection rule and evaluation code.
2. Generate the reserved holdout using scenario IDs beginning at 10,000.
3. Evaluate once.
4. Report every family, including failures.
5. Do not tune on final-holdout results.

### Step 4: Implement Variant B

If Variant A clears the quality gate:

- Cache record representations using the same weights as Variant A.
- Re-encode only records changed by an event.
- Require matching decisions and probability differences no greater than `1e-5` against full recomputation, excluding documented numeric ties.
- Measure initialization, update and query latency separately.

### Step 5: Implement Variant C

Only after Variant B is correct and benchmarked:

- Implement the learned persistent-memory updater.
- Train chronologically on event sequences.
- Test overwrites, deletions, corrections, unrelated events, missed versions and histories up to 1,024 updates.
- Compare against Variant B on quality, calibration, latency and memory footprint.
- Continue only if Variant C provides at least a two-times latency improvement or a four-times state-memory reduction while staying within the preregistered quality limits.

### Step 6: Broaden beyond synthetic tasks

If the architecture passes the controlled tests:

- Add a licensed, human-reviewed natural-language benchmark.
- Hold out an entire task family for transfer evaluation.
- Add Binary and ordered Score tasks.
- Calibrate uncertainty and abstention.
- Build the API and demo around the actual evaluated checkpoint.

## 7. Decision gates

| Gate | Required evidence | Decision if it fails |
| --- | --- | --- |
| Prediction quality | Beat the strongest non-oracle simple baseline on at least two families | Improve data/readout before memory work |
| Cache correctness | Variant B matches Variant A within tolerance | Fix cache semantics |
| Learned-memory value | Material efficiency gain with limited quality loss | Retain Variant B and stop expanding Variant C |
| Long-history reliability | Safe handling of corrections, deletes and version gaps | Restrict memory lifetime or reject Variant C |
| Transfer | Improvement on new entities, paraphrases and a withheld family | Do not claim generalization |
| Calibration | Acceptable Brier, log loss, calibration and risk/coverage | Do not expose confidence-based automation |

## 8. Current recommendation

Do not release NKE as a trained model yet. Continue with a larger, more diverse, multi-seed Variant A experiment and stronger structured baselines. Do not spend effort on learned persistent memory until the underlying decision model clears the two-family prediction gate. The next milestone is evidence that the model learns more than task-class frequencies on at least two controlled families.
