# FedSigGraph-5GC

Research prototype for **self-supervised anomaly detection in 5G Core signaling** using causal temporal graphs, protocol semantics, and later personalized federated learning.

## Current implementation scope

The first milestone focuses on a reproducible centralized baseline:

- canonical event schema for NGAP/NAS/SBI/PFCP-style signaling;
- Granomaly CSV adapter using real timestamps when available;
- strict separation between local entity identity and transferable NF role;
- causal temporal state: score an event before observing it;
- protocol-aware hard-negative generation;
- a Granomaly-style TGN baseline that can later be replaced by TemporalHEAT;
- unit tests for temporal leakage and semantic invariants.

The implementation is intentionally staged. Federated training will be added only after the centralized semantic baseline is measurable.

## Research direction

FedSigGraph-5GC models

```text
P(next endpoint role, procedure, protocol, timing | preceding 5G session context)
```

rather than only classifying individual packets or flows.

Raw entity identifiers, UE/session context, temporal memory, and anomaly calibration are intended to stay local when the federated stage is introduced.
