from fedsiggraph.data.federated_vocab import (
    FederatedSemanticVocabularies,
)


def test_same_semantic_token_has_same_id_on_independent_clients():
    client_a = FederatedSemanticVocabularies.create()
    client_b = FederatedSemanticVocabularies.create()

    token = "POST /nsmf-pdusession/v1/sm-contexts/{id}/release"
    assert client_a.procedure.encode(token) == client_b.procedure.encode(token)
    assert client_a.relation.encode("SBI::nsmf-pdusession") == (
        client_b.relation.encode("SBI::nsmf-pdusession")
    )


def test_unseen_procedure_is_hashable_without_vocabulary_fit():
    vocab = FederatedSemanticVocabularies.create()
    value = vocab.procedure.encode("vendor-specific-new-procedure")

    assert value not in (vocab.procedure.pad_id, vocab.procedure.unk_id)
    assert 0 <= value < len(vocab.procedure)


def test_fixed_roles_and_protocols_are_aligned_and_unknown_safe():
    vocab = FederatedSemanticVocabularies.create()

    assert vocab.src_role.encode("AMF") == vocab.dst_role.encode("AMF")
    assert vocab.protocol.encode("PFCP") != vocab.protocol.unk_id
    assert vocab.protocol.encode("FUTURE_PROTOCOL") == vocab.protocol.unk_id


def test_schema_config_and_sizes_are_deterministic():
    a = FederatedSemanticVocabularies.create(
        relation_buckets=64,
        procedure_buckets=128,
        message_type_buckets=32,
    )
    b = FederatedSemanticVocabularies.create(
        relation_buckets=64,
        procedure_buckets=128,
        message_type_buckets=32,
    )

    assert a.config() == b.config()
    assert a.sizes() == b.sizes()
