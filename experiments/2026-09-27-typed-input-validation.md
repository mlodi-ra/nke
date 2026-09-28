# Typed-input validation, 27 September 2026

Status: exploratory CPU experiment, not a model release. This experiment did not pass the two-family validation gate.

The first training command selected rows in family order. At 500 rows it used 365 resource, 135 access and zero priority rows. The corrected sampler selected 165 resource, 170 access and 165 priority rows, keeping each scenario's five steps together.

We then added a trainable numeric input channel, attached each candidate's current record when one exists, declared when `none` is an available choice, and evaluated on all 140 validation rows. The ModernBERT backbone remained frozen. The heads trained for two epochs on 500 selected rows with seed 17. The selected checkpoint is the second epoch by macro validation accuracy. It remains local and is not published as model weights.

| Family | Validation scenarios | Rows | Train-majority baseline | Model after epoch 1 | Model after epoch 2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Access | 12 | 60 | 36/60 (60.0%) | 36/60 (60.0%) | 36/60 (60.0%) |
| Priority | 7 | 35 | 24/35 (68.6%) | 24/35 (68.6%) | 24/35 (68.6%) |
| Resource | 9 | 45 | 15/45 (33.3%) | 26/45 (57.8%) | 39/45 (86.7%) |

The rows within each scenario are correlated. These small validation counts cannot support a generalization or statistical significance claim. The original 100-scenario test split was already examined during earlier diagnosis, so it is not a valid final test for this revision. A separate scenario range is reserved for final evaluation after model selection. It has not been evaluated.

The immediate research question is whether priority and access can learn from more diverse, balanced trajectories and a better shared query/readout. Compare stronger non-oracle baselines and multiple seeds before implementing cached or recurrent memory. Do not infer general enterprise decision capability from this generator or its oracle.
