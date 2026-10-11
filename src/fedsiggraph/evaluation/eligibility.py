"""Dataset evidence/claim eligibility derived from prepared context coverage."""

from __future__ import annotations


def research_claim_eligibility(
    context_coverage: dict[str, dict[str, float]],
    *,
    minimum_cross_protocol_events_fraction: float = 1e-9,
) -> dict[str, bool]:
    """Declare which research claims the prepared dataset can actually test.

    Cross-protocol continuation requires observed cross-protocol transitions in
    both training and test. A zero-coverage dataset may still evaluate temporal
    topology, semantic events, hard negatives, and heterogeneous aggregation,
    but cannot support a cross-protocol consistency claim.
    """
    train = context_coverage.get("train", {})
    test = context_coverage.get("test", {})

    train_cross = float(train.get("cross_protocol_transition", 0.0))
    test_cross = float(test.get("cross_protocol_transition", 0.0))
    train_sbi_cross = float(
        train.get("sbi_cross_protocol_transition", 0.0)
    )
    test_sbi_cross = float(
        test.get("sbi_cross_protocol_transition", 0.0)
    )
    train_context = float(train.get("explicit_context_history", 0.0))
    test_context = float(test.get("explicit_context_history", 0.0))

    return {
        "temporal_topology": True,
        "semantic_event_modeling": True,
        "heterogeneous_graph_modeling": True,
        "explicit_context_continuation": (
            train_context > 0.0 and test_context > 0.0
        ),
        "cross_protocol_continuation": (
            train_cross >= minimum_cross_protocol_events_fraction
            and test_cross >= minimum_cross_protocol_events_fraction
        ),
        "sbi_cross_protocol_continuation": (
            train_sbi_cross >= minimum_cross_protocol_events_fraction
            and test_sbi_cross >= minimum_cross_protocol_events_fraction
        ),
    }
