import pandas as pd
import pytest

from clairvoyance2.data import Dataset, TimeSeriesSamples
from clairvoyance2.preprocessing import ExtractTargetsTS

# pylint: disable=redefined-outer-name
# ^ Otherwise pylint trips up on pytest fixtures.


@pytest.fixture
def three_numeric_dfs():
    df_0 = pd.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0], "c": ["c1", "c2", "c3"], "d": [0, 1, 1]})
    df_1 = pd.DataFrame({"a": [7, 8, 9], "b": [7.0, 8.0, 9.0], "c": ["c1", "c1", "c3"], "d": [0, 1, 0]})
    df_2 = pd.DataFrame({"a": [3, 4, 5], "b": [3.0, 4.0, 5.0], "c": ["c2", "c2", "c2"], "d": [1, 1, 1]})
    return (df_0, df_1, df_2)


class TestIntegration:
    @pytest.mark.parametrize(
        "targets, expected_target_features, expected_covariate_features",
        [
            (["c", "d"], ["c", "d"], ["a", "b"]),
            (slice("c", "d"), ["c", "d"], ["a", "b"]),
            (slice("b", None), ["b", "c", "d"], ["a"]),
            (["b", "d"], ["b", "d"], ["a", "c"]),
            (["b"], ["b"], ["a", "c", "d"]),
            ("c", ["c"], ["a", "b", "d"]),
        ],
    )
    def test_extract_subset(self, targets, expected_target_features, expected_covariate_features, three_numeric_dfs):
        extractor = ExtractTargetsTS(params=dict(targets=targets))
        temporal_covariates = TimeSeriesSamples(three_numeric_dfs, categorical_features=["c"])
        data = Dataset(temporal_covariates=temporal_covariates, static_covariates=None, temporal_targets=None)

        data_extracted = extractor.fit_transform(data)

        assert data_extracted.temporal_covariates is not None
        assert data_extracted.temporal_targets is not None

        assert len(data_extracted.temporal_targets.feature_types) == len(expected_target_features)
        assert [f in expected_target_features for f in data_extracted.temporal_targets.feature_types]

        assert len(data_extracted.temporal_covariates.feature_types) == len(expected_covariate_features)
        assert [f in expected_covariate_features for f in data_extracted.temporal_covariates.feature_types]

    def test_extract_all(self, three_numeric_dfs):
        # NOTE: This case of no temporal covariates isn't properly handled elsewhere, needs to be handled properly.

        extractor = ExtractTargetsTS(params=dict(targets=["a", "b", "c", "d"]))
        temporal_covariates = TimeSeriesSamples(three_numeric_dfs, categorical_features=["c"])
        data = Dataset(temporal_covariates=temporal_covariates, static_covariates=None, temporal_targets=None)

        with pytest.raises(NotImplementedError) as excinfo:
            _ = extractor.fit_transform(data)
        assert "no covariates remains" in str(excinfo.value).lower()

    def test_extract_none(self, three_numeric_dfs):
        extractor = ExtractTargetsTS(params=dict(targets=[]))
        temporal_covariates = TimeSeriesSamples(three_numeric_dfs, categorical_features=["c"])
        data = Dataset(temporal_covariates=temporal_covariates, static_covariates=None, temporal_targets=None)

        data_extracted = extractor.fit_transform(data)

        assert data_extracted.temporal_covariates is not None
        assert data_extracted.temporal_targets is None

        assert len(data_extracted.temporal_covariates.feature_types) == len(["a", "b", "c", "d"])
        assert [f in ["a", "b", "c", "d"] for f in data_extracted.temporal_covariates.feature_types]
