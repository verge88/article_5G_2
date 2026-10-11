import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "inspect_origami_state_machine",
    ROOT / "scripts" / "inspect_origami_state_machine.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_state_machine_protocol_family_classification():
    assert MODULE.protocol_family("POST-nausf-auth") == "SBI"
    assert MODULE.protocol_family("200") == "SBI"
    assert MODULE.protocol_family("SessionEstablishmentRequest") == "PFCP"
    assert MODULE.protocol_family("InitialUeMessage") == "NGAP_OR_NAS"
