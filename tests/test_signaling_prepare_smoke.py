import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

import pandas as pd

from fedsiggraph.data import SignalingParserAdapter, fit_preprocessing
from fedsiggraph.data.pyg import to_temporal_data


def test_correlated_signaling_reaches_temporal_data_with_cross_protocol_mask():
    rows = []
    protocols = [
        ("gnb1", "amf", "InitialUeMessage, NASType.RegistrationRequest"),
        ("amf", "smf", "POST: /nsmf-pdusession/v1/sm-contexts"),
        ("smf", "upf", "SessionEstablishmentRequest"),
    ]
    # Enough benign events for train/val/calibration partitions.
    for cycle in range(8):
        for offset, (src, dst, message) in enumerate(protocols):
            rows.append(
                {
                    "time": f"{1 + cycle}.{offset + 1}",
                    "src": src,
                    "dst": dst,
                    "message": message,
                    "ue": f"imsi-{cycle:03d}",
                    "pdu_session_id": str(cycle),
                }
            )

    benign = SignalingParserAdapter("site-a").from_dataframe(
        pd.DataFrame(rows),
        label=0,
    )

    test_rows = pd.DataFrame(
        {
            "time": ["20.1", "20.2", "20.3"],
            "src": ["gnb1", "amf", "smf"],
            "dst": ["amf", "smf", "upf"],
            "message": [
                "InitialUeMessage, NASType.RegistrationRequest",
                "POST: /nsmf-pdusession/v1/sm-contexts",
                "SessionDeletionRequest",
            ],
            "ue": ["imsi-test"] * 3,
            "pdu_session_id": ["99"] * 3,
        }
    )
    test_events = SignalingParserAdapter("site-a").from_dataframe(
        test_rows,
        label=1,
        attack_type="synthetic-session-deletion",
    )

    fitted = fit_preprocessing(
        benign,
        test_events,
        train_fraction=0.70,
        val_fraction=0.15,
    )
    split = fitted.splits

    test_data = to_temporal_data(
        split.test,
        vocab=fitted.vocab,
        node_index=fitted.node_index,
        numeric_encoder=fitted.numeric_encoder,
        time_origin_ns=min(event.timestamp_ns for event in split.test),
        history=(),
    )

    assert test_data.context_history_mask.tolist() == [False, True, True]
    assert test_data.cross_protocol_mask.tolist() == [False, True, True]
    assert int(test_data.y.sum().item()) == 3
