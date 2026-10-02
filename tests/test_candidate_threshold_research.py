from scripts.candidate_threshold_research import split_of


def test_threshold_research_time_splits():
    assert split_of("2022-12-30") == "train"
    assert split_of("2023-01-03") == "validation"
    assert split_of("2024-12-31") == "validation"
    assert split_of("2025-01-02") == "oos"
