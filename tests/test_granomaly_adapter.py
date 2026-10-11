import pandas as pd

from fedsiggraph.data.granomaly import GranomalyAdapter


def test_adapter_uses_real_timestamp_and_stable_chronological_order():
    df = pd.DataFrame(
        {
            "ip.src": ["10.0.0.2", "10.0.0.1"],
            "ip.dst": ["10.0.0.3", "10.0.0.2"],
            "frame.time_epoch": ["2.000000001", "1.000000009"],
            "frame.protocols": ["eth:ip:udp:pfcp", "eth:ip:tcp:http2"],
            "Label": ["attack-x", "normal"],
        }
    )
    adapter = GranomalyAdapter(
        "site-a",
        role_by_entity={
            "10.0.0.1": "AMF",
            "10.0.0.2": "SMF",
            "10.0.0.3": "UPF",
        },
    )

    events = adapter.from_dataframe(df)

    assert [e.timestamp_ns for e in events] == [
        1_000_000_009,
        2_000_000_001,
    ]
    assert [e.protocol for e in events] == ["SBI", "PFCP"]
    assert events[0].label == 0
    assert events[1].label == 1
    assert events[0].src_role == "AMF"
