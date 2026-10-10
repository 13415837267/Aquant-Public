import numpy as np
import pandas as pd

from scripts.production_model import (
    DEFAULT_MODEL_CODE_COMMIT,
    DEFAULT_MODEL_PATH,
    DEFAULT_MODEL_VERSION,
    PRODUCTION_V1_FEATURES,
    load_model,
    make_production_features,
)


def test_fixed_production_model_matches_its_locked_feature_schema():
    model, payload = load_model(
        DEFAULT_MODEL_PATH,
        expected_version=DEFAULT_MODEL_VERSION,
        expected_commit=DEFAULT_MODEL_CODE_COMMIT,
    )
    assert payload["features"] == list(PRODUCTION_V1_FEATURES)
    assert model.w1.shape[0] == len(PRODUCTION_V1_FEATURES)


def test_production_features_ignore_research_only_feature_expansion():
    frame = pd.DataFrame({name: [1.0, 2.0] for name in PRODUCTION_V1_FEATURES})
    frame["market_breadth_pct"] = [20.0, 80.0]
    frame["market_median_return_pct"] = [-1.0, 1.0]
    frame["close_vs_ma5_pct"] = [1000.0, -1000.0]

    features = make_production_features(frame)

    assert features.shape == (2, len(PRODUCTION_V1_FEATURES))
    assert np.allclose(features[:, 0], [0.5, 1.0])
    assert np.allclose(features[:, -2], [0.2, 0.8])
    assert np.allclose(features[:, -1], [-0.2, 0.2])


def test_empty_input_keeps_the_locked_feature_dimension():
    empty = pd.DataFrame(columns=list(PRODUCTION_V1_FEATURES))
    assert make_production_features(empty).shape == (0, len(PRODUCTION_V1_FEATURES))
