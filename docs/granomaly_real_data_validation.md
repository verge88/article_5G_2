# Granomaly real-data validation

This note records the first end-to-end preprocessing validation of the pinned
upstream Granomaly captures against the FedSigGraph-5GC data pipeline.

## Source provenance

Repository: `TF0x42/granomaly`

Pinned revision:

```text
51df94b4e88d1c9c5e499416e4b8fee5c9b93cd7
```

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `bridge_normal.csv` | 30,530,913 | `4695889ea47e97e9f14931d0144159f6852c2ba4284add8f45f7b781cb314a34` |
| `bridge_attack_nrf.csv` | 5,471,600 | `2dfd4a78fd26797aa5b6b74e771ffb29fa0881b8cea299171ff35daf0066a63f` |
| `bridge_attack_smf.csv` | 4,315,934 | `1dd8667761ca38b871ad157c13a77bceb5469c05885fc684892d87e5ebe82e35` |

The hashes are produced by `scripts/fetch_granomaly.py` and stored in
`source_manifest.json`.

## Prepared split statistics

The same benign corpus is used for the development splits in both experiments.

| Split | NRF | SMF |
| --- | ---: | ---: |
| Train | 90,396 | 90,396 |
| Validation | 19,371 | 19,371 |
| Calibration | 19,371 | 19,371 |
| Test | 23,480 | 18,474 |

NRF test history is marked `reset`: its attack capture is not treated as a
continuous temporal continuation of the benign development capture. SMF is
marked `continuous`.

## Context coverage

| Split | NRF explicit context | NRF cross-protocol | SMF explicit context | SMF cross-protocol |
| --- | ---: | ---: | ---: | ---: |
| Train | 0.4347 | 0.0000 | 0.4347 | 0.0000 |
| Validation | 0.4394 | 0.0000 | 0.4394 | 0.0000 |
| Calibration | 0.4528 | 0.0000 | 0.4528 | 0.0000 |
| Test | 0.3943 | 0.0000 | 0.3805 | 0.0000 |

The key result is that explicit same-context history exists, but no recovered
cross-protocol transition exists in either experiment.

Therefore Granomaly can support experiments on:

- real-time temporal topology;
- semantic event encoding;
- hard-negative construction;
- heterogeneous graph aggregation;
- explicit-context continuation within the available signaling stream.

It **cannot** by itself support the proposed cross-protocol
NGAP/NAS → SBI → PFCP consistency claim. That claim requires a dataset/parser
combination exposing shared UE/PDU-session context across those protocol
boundaries.

The prepared manifest now encodes this constraint under
`claim_eligibility`, so downstream experiment tooling can distinguish an
available model component from a scientifically supportable dataset claim.

## Test unknown rates

| Feature | NRF | SMF |
| --- | ---: | ---: |
| Node identity | 0.03090 | 0.00706 |
| Destination role | 0.02828 | 0.00796 |
| Procedure | 0.02632 | 0.03361 |
| Protocol | 0.00170 | 0.00217 |

These non-zero rates are useful for inductive evaluation: the test stream does
contain entities and semantic tokens absent from the benign training
vocabulary, and the preprocessing pipeline maps them to reserved unknown IDs
rather than fitting on test data.

## Validation status

The real-data GitHub Actions smoke test successfully:

1. downloaded the pinned captures;
2. verified their hashes;
3. prepared NRF and SMF data;
4. created PyG temporal tensors and evaluation metadata;
5. validated the resulting manifests;
6. uploaded preprocessing/manifests as workflow artifacts.

No attack labels are used for vocabulary fitting, numerical scaling, model
training, or threshold calibration.
