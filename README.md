# FedSigGraph-5GC

Research prototype for **self-supervised anomaly detection in 5G Core signaling** using causal temporal graphs, protocol semantics, and later personalized federated learning.

## Current milestone

The repository now contains a reproducible centralized TGN baseline and a leakage-safe preprocessing pipeline:

- canonical event schema for NGAP/NAS/SBI/PFCP signaling;
- Granomaly adapter using real `frame.time_epoch` timestamps;
- semantic extraction from `combined_info`:
  - HTTP/2 SBI method, normalized API path, status, and stream correlation;
  - PFCP establishment/modification/deletion/heartbeat procedures;
  - NGAP procedures and embedded NAS messages;
- published Granomaly QEMU IP → NF-role mapping, with user overrides;
- strict separation between deployment-local identity and transferable role;
- train-only categorical vocabulary and robust numeric scaling;
- chronological train / validation / calibration / test separation;
- causal temporal state: **score first, observe second**;
- protocol-aware hard-negative primitives;
- Granomaly-style causal TGN baseline;
- calibration threshold selected only from benign calibration traffic;
- `frozen`, `safe`, and `blind` test-memory policies;
- unit tests for temporal leakage, preprocessing leakage, and semantic parsing;
- lightweight CI plus a CPU PyTorch/PyG smoke workflow.

Federated training is intentionally deferred until the centralized semantic baseline is measurable.

## Core research hypothesis

FedSigGraph-5GC aims to model

```text
P(next endpoint role, procedure, protocol, timing | preceding 5G context)
```

instead of classifying isolated packets or flow statistics.

The intended federated stage will keep raw entity identifiers, UE/session context, temporal memory, and anomaly calibration local while sharing only trainable protocol-level parameters.

## Installation

Core preprocessing and tests:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

TGN experiments:

```bash
pip install -e ".[ml,dev]"
```

PyTorch Geometric 2.8 is supported by the current smoke test.

## Granomaly preprocessing

Obtain the upstream Granomaly captures:

```text
bridge_normal.csv
bridge_attack_nrf.csv
bridge_attack_smf.csv
```

Prepare one experiment:

```bash
python scripts/prepare_granomaly.py \
  --normal /path/to/bridge_normal.csv \
  --attack /path/to/bridge_attack_smf.csv \
  --output data/granomaly-smf
```

For the published QEMU dataset, known NF roles are inferred automatically:

```text
10.211.55.4  AMF
10.211.55.5  SMF
10.211.55.6  UPF
10.211.55.7  NRF
10.211.55.8  UDM
10.211.55.9  AUSF
10.211.55.10 UDR
10.211.55.11 SCP
10.211.55.12 PCF
10.211.55.13 BSF
10.211.55.14 NSSF
10.211.55.2  gNB
10.211.55.3  UE
```

A custom JSON mapping can override these entries:

```bash
python scripts/prepare_granomaly.py \
  --normal ... \
  --attack ... \
  --role-map roles.json \
  --output ...
```

The preparation step creates:

```text
train.pt
val.pt
calibration.pt
test.pt
preprocessing.json
manifest.json
```

All learned preprocessing state is fitted on **benign train only**. Unseen procedures, roles, and strict-inductive node identities map to dedicated unknown IDs.

If the attack capture is from a different timestamp epoch than the benign capture, `manifest.json` marks `test_history_mode=reset` so TGN memory is not artificially connected across unrelated captures.

## Baseline TGN

Run:

```bash
python scripts/train_baseline.py \
  --data data/granomaly-smf \
  --epochs 20 \
  --batch-size 512 \
  --test-memory safe \
  --output results/granomaly-smf-tgn.json
```

The model is trained as a one-class temporal link detector:

- observed benign edge → target 0;
- sampled structural negative → target 1;
- only benign observed edges enter TGN memory during training.

Threshold selection uses only benign `calibration.pt`.

### Test memory modes

`frozen`

: Test events never enter temporal memory. Useful for pure generalization measurement.

`blind`

: Every test event enters memory. Closest to ordinary online TGN operation, but vulnerable to attack-state contamination.

`safe`

: An event is observed only when its anomaly score is below the calibrated alert threshold. This is the default contamination-resistant online mode.

## Why this differs from the original Granomaly pipeline

The upstream baseline uses the row index as time, factorized `frame.protocols` as the message feature, and independently permuted source/destination endpoints as negatives.

This implementation instead preserves real timing and extracts stable 5G signaling semantics from the existing Granomaly CSV fields. The next experiment will therefore compare, in order:

```text
Granomaly-style topology TGN
        ↓
real-time numeric TGN
        ↓
semantic event encoder
        ↓
protocol-aware hard negatives
        ↓
TemporalHEAT
        ↓
cross-protocol objectives
        ↓
personalized federation
```

That staged design makes it possible to attribute gains to representation, negative construction, heterogeneous graph modeling, and federation separately.


## Staged ablation

Run the full five-stage comparison with identical seeds and evaluation:

```bash
python scripts/run_ablation.py \
  --data data/granomaly-smf \
  --output-dir results/granomaly-smf \
  --seeds 1 2 3 \
  --epochs 20 \
  --test-memory safe
```

The launcher evaluates:

```text
numeric_tgn
semantic_tgn
temporal_heat
temporal_heat_aux
temporal_heat_continuation
```

where the last three isolate heterogeneous aggregation, auxiliary continuation
training, and continuation surprise in the final anomaly score. Individual
seed JSON files are retained and `summary.json` reports mean/std overall and
per attack family.

### Context coverage is a dataset validity check

`manifest.json` reports, for each split:

```json
{
  "context_coverage": {
    "train": {
      "explicit_context_history": 0.0,
      "cross_protocol_transition": 0.0
    }
  }
}
```

These values are not model metrics. They describe whether the source capture
contains enough explicit UE/session correlation to evaluate the continuation
hypothesis. Cross-protocol loss never invents a PFCP↔SBI or NGAP↔SBI session
link from temporal proximity alone.

If `cross_protocol_transition` is approximately zero, the dataset remains
valid for topology, semantic-event, hard-negative, and heterogeneous-GNN
experiments, but it must **not** be used as evidence for the cross-protocol
contribution. A richer parser/dataset exposing shared UE/PDU-session
identifiers is required for that experiment.


## Semantic TGN and TemporalHEAT

After preprocessing, the progressively stronger centralized models can be run
individually:

```bash
python scripts/train_semantic_tgn.py \
  --data data/granomaly-smf \
  --epochs 20 \
  --test-memory safe \
  --output results/granomaly-smf-semantic-tgn.json

python scripts/train_temporal_heat.py \
  --data data/granomaly-smf \
  --epochs 20 \
  --test-memory safe \
  --output results/granomaly-smf-temporal-heat.json
```

Both semantic trainers report the fraction of training events for which a
non-trivial train-derived endpoint/procedure hard negative was available.
Endpoint negatives reuse a real train-known entity of an incompatible role;
synthetic or unknown node IDs are not used as shortcuts.

Dataset-level time metadata is stored in `manifest.json`, while
`TemporalData` contains only event-aligned tensors. This is required for
correct batching with `TemporalDataLoader`.

## Framework-independent federated training

Prepare every deployment with the federated preparation script so all clients
share the same semantic schema:

```bash
python scripts/prepare_federated_client.py ...
```

Then run FedAvg or FedProx without introducing an orchestration dependency:

```bash
python scripts/train_federated.py \
  --clients data/site-a data/site-b data/site-c \
  --strategy fedprox \
  --prox-mu 0.01 \
  --rounds 10 \
  --local-epochs 1 \
  --output results/federated-fedprox.json
```

The coordinator validates the shared semantic-schema fingerprint before
training. Only trainable shared model parameters are aggregated. TGN memory,
buffers, identity mappings, local calibration state, and parameters under the
`local_adapter.*` namespace remain private.

Federated TemporalHEAT enables a small residual personalization adapter by
default (`--local-adapter-dim 16`). Its upper projection is zero-initialized,
so the adapter starts as an identity mapping and learns deployment-specific
corrections locally. The adapter persists across rounds while the local Adam
optimizer state is reset at every round boundary. Set
`--local-adapter-dim 0` for a non-personalized FedAvg/FedProx ablation.

After the final round, each client performs its own benign calibration and test
evaluation. Thresholds are therefore never averaged across deployments.

The framework-independent coordinator is intentionally the reference
implementation; a Flower adapter can wrap the same parameter-partition and
aggregation functions later.



## Federated robustness matrix

Controlled FL robustness experiments can be launched without changing the
training/evaluation code:

```bash
python scripts/run_federated_robustness.py \
  --clients data/site-a data/site-b data/site-c \
  --poison-client site-c \
  --poison-mode sign_flip \
  --poison-factor 5 \
  --rounds 10 \
  --seeds 1 2 3 \
  --output-dir results/federated-robustness
```

The matrix always includes clean FedAvg and clean FedProx. When a poison client
is provided it additionally compares poisoned FedProx with weighted mean,
coordinate-wise median, and trimmed mean aggregation. Poisoning operates only on
the selected client's outgoing model update relative to the current global
parameters; it does not modify packet traces or ground-truth labels.

This mode is intended only for controlled resilience experiments. The generated
`summary.json` reports mean client PR-AUC, recall, and false-positive rate for
each strategy/aggregation condition while retaining every per-seed result file.


## Evaluation-only attack metadata

Preparation also writes `test_metadata.csv`. It contains the original attack
family solely for final reporting. It is excluded from vocabulary fitting,
numeric scaling, model input, training loss, memory updates, and threshold
calibration. Trainers verify that its binary labels align exactly with
`test.pt` before producing per-attack recall.
