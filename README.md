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
