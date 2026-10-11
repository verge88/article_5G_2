from fedsiggraph.evaluation.eligibility import research_claim_eligibility


def test_zero_cross_protocol_coverage_blocks_cross_protocol_claim_only():
    eligibility = research_claim_eligibility(
        {
            "train": {
                "explicit_context_history": 0.4,
                "cross_protocol_transition": 0.0,
            },
            "test": {
                "explicit_context_history": 0.3,
                "cross_protocol_transition": 0.0,
            },
        }
    )

    assert eligibility["temporal_topology"] is True
    assert eligibility["semantic_event_modeling"] is True
    assert eligibility["heterogeneous_graph_modeling"] is True
    assert eligibility["explicit_context_continuation"] is True
    assert eligibility["cross_protocol_continuation"] is False
    assert eligibility["sbi_cross_protocol_continuation"] is False


def test_cross_protocol_claim_requires_train_and_test_coverage():
    eligibility = research_claim_eligibility(
        {
            "train": {
                "explicit_context_history": 0.2,
                "cross_protocol_transition": 0.1,
            },
            "test": {
                "explicit_context_history": 0.2,
                "cross_protocol_transition": 0.05,
            },
        }
    )

    assert eligibility["cross_protocol_continuation"] is True
    assert eligibility["sbi_cross_protocol_continuation"] is False


def test_primary_sbi_cross_protocol_claim_requires_sbi_transitions_in_both_splits():
    eligibility = research_claim_eligibility(
        {
            "train": {
                "explicit_context_history": 0.5,
                "cross_protocol_transition": 0.2,
                "sbi_cross_protocol_transition": 0.1,
            },
            "test": {
                "explicit_context_history": 0.4,
                "cross_protocol_transition": 0.1,
                "sbi_cross_protocol_transition": 0.05,
            },
        }
    )

    assert eligibility["cross_protocol_continuation"] is True
    assert eligibility["sbi_cross_protocol_continuation"] is True
