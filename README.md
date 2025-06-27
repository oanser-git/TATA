# TATA: Benchmark NIDS Test Sets Assessment and Targeted Augmentation

> This repository accompanies the paper **“TATA: Benchmark NIDS Test Sets
> Assessment and Targeted Augmentation,”** accepted in the **Main Track** of
> *ESORICS 2025*.
>
> **Notice (June 2025)** — We are currently restructuring and cleaning the
> codebase. A fully reproducible, well-documented release will follow shortly.
>
> The **augmented test set** will be released simultaneously with the cleaned
> codebase; a download link will be provided here.

**What it is.** TATA (Test-sets Assessment and Targeted Augmentation) is a
**model-agnostic framework for evaluating and improving benchmark
Network-Intrusion-Detection-System (NIDS) test sets**. It **assesses** how well
a test set stresses a NIDS by comparing it to the corresponding training set
through three complementary, dataset-centric metrics (**diversity, proximity,
and scarcity**) and then closes the uncovered gaps by generating *real* network
flows that make future evaluations more challenging.

**How it works.** The pipeline (i) trains a **contrastive auto-encoder** on the
training split to construct a structured latent space in which flows cluster by
traffic type; (ii) embeds the original test split into that space to compute
the three quality metrics cluster-wise; and (iii) employs an **offline
deep-reinforcement-learning agent** (CQL / TD3+BC) that directs a configurable
traffic testbed (changing loss, jitter, delay, etc.) to produce additional
flows that maximise diversity, proximity, and scarcity scores.
