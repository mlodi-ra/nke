# NKE: Architecture decisions and experiment specification

**Date:** 27 September 2026

**Status:** Design frozen for initial implementation; no performance results yet.
**Relationship:** Technical specification for the NEXUS model-first plan. This specification moves the state-update experiment into the first research cycle, before a public launch.

## 1. Research objective

Build a trained model that answers typed decision questions from changing state, with reusable representations, dynamic candidates, and measurable uncertainty. The central hypothesis is that retaining and updating learned state can reduce repeated computation while preserving decision quality on new questions.

Two corrections to our earlier framing matter:

- NKE still needs observations. It cannot discover missing enterprise facts or eliminate the need for event feeds. Learning a representation does not solve data acquisition.
- Caching previously encoded records is useful engineering. A novel learned memory mechanism must improve on that baseline before we attribute an advantage to the architecture.

Cross-attention, latent memory, and typed prediction heads have substantial prior art, including Perceiver IO. Any future novelty claim must identify our specific contribution and undergo a broader literature review. [1]

## 2. Architecture decisions

| Decision | Initial choice | Reason and consequence |
| --- | --- | --- |
| Backbone | ModernBERT-base, 149M pretrained parameters; pin its revision before training | Licensed Apache 2.0, encoder architecture and published weights. This is a starting candidate, not a claim it is optimal. [2] |
| Observations | Versioned records with entity ID, typed fields, relationships, timestamp and update/delete operation | Supports explicit corrections and reproducible state. Text fields use tokenization; numeric fields retain typed values and units. |
| Query | Question text, output type, optional rubric, candidate IDs/descriptions and eligibility mask | Decisions are conditioned on the question and current candidates rather than fixed class IDs. |
| Record representation | Independently encode each record; learned pooling to four 256-dimensional tokens per record | Makes unchanged records reusable. Independent encodings cannot capture all cross-record relations; query attention must recover them. |
| Query network | Question/candidate encodings projected to 256 dimensions, two cross-attention blocks, four heads | Candidate queries attend to state memory in parallel. This avoids sequential output decoding but still incurs computation per candidate. |
| Choice | Shared scalar scorer applied to each candidate query; masked softmax | Permutation-equivariant candidate scoring; no candidate-position embedding. |
| Binary | Proposition-conditioned logit and sigmoid | Direct learned probability, separately calibrated. |
| Score | Distribution over five explicitly described ordered rubric levels; expectation returned | Avoids pretending that an arbitrary numeric scale has intrinsic meaning. No exact error-bound claim. |
| Output dependencies | Questions initially evaluated independently against common memory | Multiple outputs can be mutually inconsistent. Jointly constrained graphs/actions need separate research. |
| Persistent learned memory | Experimental alternative with 64 latent slots of width 256; gated cross-attention updates from changed records | Must outperform record caching at comparable quality. Deletion, overwrite and long-history behavior are core tests. |

Initial limits: 64 records, at most 128 text tokens per record, 2–32 candidates, and 1–8 questions per call. Overflow is a typed error rather than silent truncation. Stress tests extend to 256 records, 128 candidates and longer event histories after the initial gate. These are experiment settings, not product limits or advertised capabilities.

The learned-memory updater sees the old and new record encodings for replacement/deletion events. A deterministic event store retains the authoritative state for reset and replay. It is infrastructure around the model, not the decision mechanism.

## 3. Three variants that isolate the research claim

**A. Full recomputation:** Re-encode every current record and run the query network on each request. This is the clean reference for the record architecture. Also include a joint snapshot encoder on examples fitting its context limit to expose quality lost by independent record encoding.

**B. Cached records:** Use identical weights and query network as A; replace only changed record encodings. Same current records must yield matching predictions within numeric tolerance. This establishes the gain from caching alone.

**C. Learned persistent memory:** Replace the full record-token memory with the 64-slot learned memory. Train its updater and readout on event sequences. Compare against B on speed, memory use, quality, calibration and correction behavior. Include an experiment rebuilding the same 64-slot memory from the full snapshot to isolate compression effects from recurrent-update effects.

Cached representations carry the model/tokenizer revision. Updating model weights invalidates all caches. Time-dependent age features are refreshed at query time or explicitly invalidated, not frozen in a cached embedding. A version gap, out-of-order update, or unresolvable entity reference triggers resynchronization or a typed abstention. Silent reuse of suspect memory fails the engineering gate.

## 4. API and probability semantics

Expose `initialize(snapshot)`, `update(events)` and `decide(questions, state_version)`. A response includes model/state version, typed predictions, raw probabilities and abstention disposition. The model never directly executes actions.

Choice probabilities describe relative suitability among supplied candidates. A separate learned answerability signal supports abstention when no candidate is supported. Calibrate and evaluate both; neither supplies a universal safety guarantee.

An eligibility mask removes disallowed options before selection. Zero eligible options returns abstention without softmax. One eligible option has conditional probability 1 by construction, which is **not** 100% confidence that the option is correct. Preserve answerability and report this distinction. Include masks and changing candidate sets in calibration data. A supplied mask enforces only the exclusions it actually contains.

Cache state encodings independently of questions. Cache candidate encodings only when description, preprocessing and model revision match. If answers change materially when equivalent candidate order changes, investigate before any release.

## 5. Data and training

Start with three controlled task families that exercise different state relations: resource assignment with eligibility and capacity changes; access decisions from declared roles and policies; prioritization from dependency and urgency updates. These are simulated tasks with independently checkable answers, not live security or control applications.

Create paired snapshots/event streams, questions, candidates and labels. Include deletes, corrections, equal-score ties, missing observations, distractors and questions whose answers change after one event. Use an exact reference evaluator plus separately authored invariants and a reviewed sample to check the generator. A symbolic evaluator can solve these simulated tasks perfectly; model success here establishes architectural behavior, not superiority over rules or enterprise intelligence.

Initial data budget: 30,000 independent training trajectories, 3,000 validation, 3,000 calibration and 6,000 final test trajectories, balanced across families; 8–32 events and multiple questions per trajectory. Adjust training volume based on learning curves, not test results. Split entity sets, policy templates and scenario seeds before generating examples. Bootstrap uncertainty by trajectory, not correlated individual questions.

First train A/B readout heads with the backbone frozen. Then fine-tune selected encoder layers if held-out validation shows inadequate quality. Training cannot reuse stale embeddings after encoder weights change. Train C on chronological sequences, using supervised final targets and optional distribution matching to the trained snapshot model. Recompute calibration after changing architecture or quantization. Log weights, data hashes, seed, optimizer, learning rate and checkpoint selection.

Use cross-entropy for Choice/Binary and ordinal rubric distributions initially. Assess Brier score and log loss; add a Brier training loss only as a controlled ablation. Do not call this RLCD. Use three training seeds for shortlisted variants. A small 1,000-trajectory run is only a plumbing check, never headline evidence.

Before a general-purpose release, add a separately reviewed natural-language benchmark with licensed human-labeled examples and hold out an entire task family during transfer evaluation. Selection of that dataset remains a release dependency. Synthetic labels and teacher agreement alone cannot establish broad model intelligence.

## 6. Preregistered experiments and gates

The numbers below are proposed engineering thresholds, chosen before seeing results. They are not performance predictions.

| Experiment | Comparison | Gate or decision |
| --- | --- | --- |
| E1: Useful prediction | Learned A against majority, lexical and classical learned baselines, plus symbolic oracle | Beat the strongest non-oracle simple baseline with a paired 95% confidence interval excluding zero on at least two families; disclose every family. Otherwise improve data/model before optimizing. |
| E2: Caching correctness | A vs B, same weights and current state | Probability difference <=1e-5 on a deterministic reference implementation and same decisions, except documented numeric ties. Failures indicate cache semantics bugs. |
| E3: Learned memory value | C vs B and rebuilt compact memory | At least 2x lower p95 update-plus-query latency OR 4x lower state-memory footprint, with <=1 percentage point accuracy loss per family and <=0.01 Brier degradation. Account for raw state storage separately. |
| E4: Long histories | Deletes, overwrites, unrelated-event insertions, missed events, and histories to 1,024 updates | Report quality versus sequence length and fresh rebuild. Refuse unsupported state versions. Failure restricts memory lifetime or rejects C. |
| E5: Decision flexibility | New entities, candidate descriptions, question paraphrases and a withheld task family | Compare with strongest baseline; publish separate in-distribution and transfer results. No transfer claim if improvements disappear on the withheld family. |
| E6: Uncertainty | Untouched calibration/test split; distribution shift and masks | Report Brier, log loss, 15-bin equal-width ECE and risk/coverage curves. Pick thresholds on calibration only. Report accepted-set error and intervals, never ECE alone. |

If C fails E3, stop expanding its architecture until we diagnose the loss. B can support a useful trained model release, but the claim of an architectural memory advantage remains unproven. If C passes only by compressing away facts needed for new questions, it fails the intended use even if average metrics look good.

## 7. Benchmark integrity

Benchmark state sizes 16/64/256 records; changed fractions 1/10/100%; question counts 1/8/32; and candidate counts 4/32/128 as supported. Report batch-one interactive latency separately from batch throughput. Include cold initialization, candidate encoding, update cost, full query path, resynchronizations and device transfers. Amortize initialization over reported query counts. Memory benchmarks include caches and model weights, with persistent raw state disclosed separately.

Use matched hardware, precision, workloads and quality constraints. Record CPU/GPU model, versions, warm-up, request count and raw timings. Use at least 10,000 timed calls for a final p99 report; stop exploratory timing much earlier. Ablations should resolve the four material questions: does caching suffice, does compression lose facts, does recurrence accumulate error, and does the model generalize to new questions?

No Jev comparison until we can call it on comparable tasks. Local compute latency and a remote service round-trip are different measurement boundaries. Publish both when available.

## 8. Implementation handoff and compute

Build in this order: event/state contract and exact replay; data generator/oracle and fixed split manifest; A with trainable heads; B with cache equivalence checks; C with sequence training; benchmark/calibration harness; API and demo using the actual checkpoint. Begin with Choice and Binary; add the rubric head after they train successfully.

The present workspace reports nine logical CPUs, no installed PyTorch, and no `nvidia-smi` command. This does not establish an available CUDA training device. It is sufficient for specifications and code development; real training requires a confirmed compute environment and model-download access. Plan around one 24–48 GB CUDA GPU initially, profile a small run, and estimate time/cost from measured throughput before authorizing longer paid runs. Actual fit depends on batching and which encoder layers are trainable. No paid training or model download has occurred in this task.

Completion of this task means the architecture and falsification experiments are specified. Implementation belongs in Sol Medium. Return to Astra for failed research gates, ambiguous results or major architecture changes. The first implementation deliverable is a working data/replay pipeline and a trainable A model, followed by a saved checkpoint when compute is available.

## Sources

1. [Perceiver IO: A General Architecture for Structured Inputs & Outputs](https://arxiv.org/abs/2107.14795). Prior art for flexible querying and latent cross-attention, not evidence that NKE's proposed update mechanism works.
2. [ModernBERT-base model card](https://huggingface.co/answerdotai/ModernBERT-base). Published encoder, parameter count, context and license. Accessed 27 September 2026.
