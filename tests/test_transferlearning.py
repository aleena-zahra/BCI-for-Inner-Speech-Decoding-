import pytest

from voice.evaluation.transferlearning import evaluate_concept_mapped_transfer


def test_transfer_scores_only_after_explicit_semantic_mapping():
    result = evaluate_concept_mapped_transfer(
        ["palabra_uno", "palabra_dos", "palabra_uno"],
        [10, 20, 20],
        target_to_concept={"palabra_uno": "yes", "palabra_dos": "no"},
        source_class_to_concept={10: "yes", 20: "no"},
    )
    assert result["concept_mapped_accuracy"] == pytest.approx(2 / 3)
    assert result["n_trials"] == 3
    assert result["per_concept_accuracy"] == {"no": 1.0, "yes": 0.5}


def test_transfer_rejects_missing_label_mapping():
    with pytest.raises(ValueError, match="no concept mapping"):
        evaluate_concept_mapped_transfer(
            ["class-a", "class-b"], [0, 1],
            target_to_concept={"class-a": "concept-a"},
            source_class_to_concept={0: "concept-a", 1: "concept-b"},
        )


def test_transfer_rejects_ambiguous_mapping():
    with pytest.raises(ValueError, match="one-to-one"):
        evaluate_concept_mapped_transfer(
            ["a", "b"], [0, 1],
            target_to_concept={"a": "same", "b": "same"},
            source_class_to_concept={0: "same", 1: "other"},
        )
