import numpy as np

from scripts.train_pairwise_mlp_ranker import PairwiseMLPRankingModel


def test_pairwise_mlp_update_improves_positive_negative_ordering():
    features = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    labels = np.asarray([1.0, 0.0], dtype=np.float64)
    model = PairwiseMLPRankingModel(
        n_features=2,
        hidden_units=8,
        learning_rate=0.1,
        l2=0.0,
        max_positives_per_day=1,
        max_negatives_per_day=1,
        seed=7,
    )

    before = float(model.predict(features)[0] - model.predict(features)[1])
    pairs = model.update(features, labels)
    after = float(model.predict(features)[0] - model.predict(features)[1])

    assert pairs == 1
    assert model.pair_updates == 1
    assert model.pairs_seen == 1
    assert after > before


def test_pairwise_mlp_update_skips_days_without_both_classes():
    features = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    model = PairwiseMLPRankingModel(n_features=2, hidden_units=4, seed=7)

    assert model.update(features, np.asarray([1.0, 1.0])) == 0
    assert model.update(features, np.asarray([0.0, 0.0])) == 0
    assert model.pair_updates == 0
    assert model.pairs_seen == 0


def test_pairwise_mlp_rejects_mismatched_features():
    model = PairwiseMLPRankingModel(n_features=2, hidden_units=4, seed=7)

    try:
        model.predict(np.asarray([[1.0, 0.0, 2.0]], dtype=np.float64))
    except ValueError as error:
        assert "特征数量" in str(error)
    else:
        raise AssertionError("特征维度不匹配时应抛出 ValueError")
