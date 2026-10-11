from fedsiggraph.evaluation.provenance import runtime_provenance


def test_runtime_provenance_contains_reproducible_software_metadata_only():
    value = runtime_provenance()

    assert "repository_commit" in value
    assert "software_versions" in value
    assert "python" in value["software_versions"]
    assert "hostname" not in value
    assert "username" not in value
