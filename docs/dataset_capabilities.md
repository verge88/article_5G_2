# Empirically verified dataset capabilities

This note records what the current FedSigGraph extraction pipeline actually
observed in external captures. It is intentionally stricter than dataset
descriptions: a protocol is marked available only when the reproducible CI
pipeline extracted it.

## ORIGAMI 5G Signaling Traffic — example raw PCAP

Source: Zenodo record 19605370, `example.pcap.gz`.

Verified by the `origami-data-sanity` workflow using the causal PCAP exporter.

Observed on the first successful real-data run:

- 22,575 signaling events;
- 14,311 NGAP events;
- 8,264 PFCP events;
- 0 SBI events from automatic packet dissection;
- 22,319 events with explicit UE context;
- 8,012 events with explicit session context;
- 21,319 events (94.44%) with preceding explicit context;
- 996 generic cross-protocol transitions (4.41%);
- the observed transition split was dominated by PFCP → NGAP;
- raw-PCAP SBI continuation is therefore **not claim-eligible** yet.

The dataset remains useful for N2/N4 temporal-graph and context-correlation
experiments. The absence of extracted SBI must not be interpreted as absence of
SBI traffic in the dataset; the published description states that SBI traffic
is present, typically over HTTP/2/TCP/TLS. It means the current causal packet
extractor does not yet expose those SBI events from this capture.

## DLTeamTUC Open5GS fuzzing PCAPs

Source repository revision:
`DLTeamTUC/5GDatasets@e71267ca84f6f38ab39f830a2c515555bb5f5898`.

Seven small Open5GS fuzzing captures were processed by the
`dlteam-data-sanity` workflow.

Observed:

- 90 NGAP events total;
- 0 PFCP events;
- 0 SBI events;
- 89/90 events (98.89%) with preceding explicit UE context;
- 0 cross-protocol transitions;
- inferred topology: gNB ↔ AMF, with no unknown endpoint roles.

These captures are suitable as a small N2/NAS malformed-signaling sanity set,
not as evidence for N4 or SBI cross-protocol anomaly detection.

## Claim policy

The project distinguishes:

- `cross_protocol_continuation`: any explicit protocol transition;
- `sbi_cross_protocol_continuation`: transitions involving SBI.

The primary NGAP/NAS ↔ SBI ↔ PFCP research claim requires the second condition
to be non-zero in both training and test. Datasets with only NGAP↔PFCP or only
NGAP are still valid for their narrower ablations, but cannot unlock the
primary claim.

All counts above should be regenerated from CI artifacts before publication,
rather than copied blindly from this note.
