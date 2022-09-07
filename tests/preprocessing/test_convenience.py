import pandas as pd
import pytest

from clairvoyance2.data import Dataset, StaticSamples, TimeSeriesSamples
from clairvoyance2.preprocessing import (
    StaticFeaturesConcatenator,
    TemporalTargetsExtractor,
    TemporalTreatmentsExtractor,
    TimeIndexFeatureConcatenator,
)

# pylint: disable=redefined-outer-name
# ^ Otherwise pylint trips up on pytest fixtures.


@pytest.fixture
def three_mixed_dfs():
    df_0 = pd.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0], "c": ["c1", "c2", "c3"], "d": [0, 1, 1]})
    df_1 = pd.DataFrame({"a": [7, 8, 9], "b": [7.0, 8.0, 9.0], "c": ["c1", "c1", "c3"], "d": [0, 1, 0]})
    df_2 = pd.DataFrame({"a": [3, 4, 5], "b": [3.0, 4.0, 5.0], "c": ["c2", "c2", "c2"], "d": [1, 1, 1]})
    return (df_0, df_1, df_2)


class TestIntegration:
    class TestExtractTargetsTC:
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
        def test_extract_subset(self, targets, expected_target_features, expected_covariate_features, three_mixed_dfs):
            extractor = TemporalTargetsExtractor(params=dict(targets=targets))
            temporal_covariates = TimeSeriesSamples(three_mixed_dfs)
            data = Dataset(temporal_covariates=temporal_covariates, static_covariates=None, temporal_targets=None)

            data_extracted = extractor.fit_transform(data)

            assert data_extracted.temporal_covariates is not None
            assert data_extracted.temporal_targets is not None

            assert len(data_extracted.temporal_targets.features) == len(expected_target_features)
            assert [f in expected_target_features for f in data_extracted.temporal_targets.features.keys()]

            assert len(data_extracted.temporal_covariates.features) == len(expected_covariate_features)
            assert [f in expected_covariate_features for f in data_extracted.temporal_covariates.features.keys()]

        def test_extract_all(self, three_mixed_dfs):
            # NOTE: This case of no temporal covariates isn't properly handled elsewhere, needs to be handled properly.

            extractor = TemporalTargetsExtractor(params=dict(targets=["a", "b", "c", "d"]))
            temporal_covariates = TimeSeriesSamples(three_mixed_dfs)
            data = Dataset(temporal_covariates=temporal_covariates, static_covariates=None, temporal_targets=None)

            with pytest.raises(NotImplementedError) as excinfo:
                _ = extractor.fit_transform(data)
            assert "no covariates remain" in str(excinfo.value).lower()

        def test_extract_none(self, three_mixed_dfs):
            extractor = TemporalTargetsExtractor(params=dict(targets=[]))
            temporal_covariates = TimeSeriesSamples(three_mixed_dfs)
            data = Dataset(temporal_covariates=temporal_covariates, static_covariates=None, temporal_targets=None)

            data_extracted = extractor.fit_transform(data)

            assert data_extracted.temporal_covariates is not None
            assert data_extracted.temporal_targets is None

            assert len(data_extracted.temporal_covariates.features) == len(["a", "b", "c", "d"])
            assert [f in ["a", "b", "c", "d"] for f in data_extracted.temporal_covariates.features.keys()]

    class TestExtractTreatmentsTC:
        @pytest.mark.parametrize(
            "treatments, expected_treatment_features, expected_covariate_features",
            [
                (["c", "d"], ["c", "d"], ["a", "b"]),
                (slice("c", "d"), ["c", "d"], ["a", "b"]),
                (slice("b", None), ["b", "c", "d"], ["a"]),
                (["b", "d"], ["b", "d"], ["a", "c"]),
                (["b"], ["b"], ["a", "c", "d"]),
                ("c", ["c"], ["a", "b", "d"]),
            ],
        )
        def test_extract_subset(
            self, treatments, expected_treatment_features, expected_covariate_features, three_mixed_dfs
        ):
            extractor = TemporalTreatmentsExtractor(params=dict(treatments=treatments))
            temporal_covariates = TimeSeriesSamples(three_mixed_dfs)
            data = Dataset(temporal_covariates=temporal_covariates, static_covariates=None, temporal_treatments=None)

            data_extracted = extractor.fit_transform(data)

            assert data_extracted.temporal_covariates is not None
            assert data_extracted.temporal_treatments is not None

            assert len(data_extracted.temporal_treatments.features) == len(expected_treatment_features)
            assert [f in expected_treatment_features for f in data_extracted.temporal_treatments.features.keys()]

            assert len(data_extracted.temporal_covariates.features) == len(expected_covariate_features)
            assert [f in expected_covariate_features for f in data_extracted.temporal_covariates.features.keys()]

    class TestAddTimeIndexFeatureTC:
        @pytest.mark.parametrize(
            "params, expected_time_delta_sample_0, expected_time_delta_sample_1, "
            "expected_time_index_sample_0, expected_time_index_sample_1",
            [
                (
                    # params:
                    dict(
                        add_time_index=True,
                        add_time_delta=True,
                        time_delta_pad_at_back=False,
                        time_delta_pad_value=-999.0,
                    ),
                    # expected_time_delta_sample_0:
                    [-999.0, 1, 1],
                    # expected_time_delta_sample_1@
                    [-999.0, 20, 5, 65],
                    # expected_time_index_sample_0:
                    [1, 2, 3],
                    # expected_time_index_sample_1:
                    [10, 30, 35, 100],
                ),
                (
                    dict(add_time_index=True, add_time_delta=False),
                    None,
                    None,
                    [1, 2, 3],
                    [10, 30, 35, 100],
                ),
                (
                    dict(
                        add_time_index=False,
                        add_time_delta=True,
                        time_delta_pad_at_back=False,
                        time_delta_pad_value=-999.0,
                    ),
                    [-999.0, 1, 1],
                    [-999.0, 20, 5, 65],
                    None,
                    None,
                ),
                (
                    dict(
                        add_time_index=True,
                        add_time_delta=True,
                        time_delta_pad_at_back=True,
                        time_delta_pad_value=777.0,
                    ),
                    [1, 1, 777],
                    [20, 5, 65, 777],
                    [1, 2, 3],
                    [10, 30, 35, 100],
                ),
            ],
        )
        def test_successful_transform(
            self,
            params,
            expected_time_delta_sample_0,
            expected_time_delta_sample_1,
            expected_time_index_sample_0,
            expected_time_index_sample_1,
        ):
            # Arrange:
            df_0 = pd.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]}, index=[1, 2, 3])
            df_1 = pd.DataFrame({"a": [7, 8, 9, 10], "b": [7.0, 8.0, 9.0, 10.0]}, index=[10, 30, 35, 100])
            data = Dataset(temporal_covariates=TimeSeriesSamples([df_0, df_1]))
            transformer = TimeIndexFeatureConcatenator(params=params)

            # Act:
            data = transformer.fit_transform(data)

            # Assert:
            assert data.n_samples == 2
            assert "a" in data.temporal_covariates.features and "b" in data.temporal_covariates.features

            if params["add_time_index"] is True:
                assert "time_index" in data.temporal_covariates.features
                assert data.temporal_covariates.features["time_index"].numeric_compatible
                for ts in data.temporal_covariates:
                    assert "time_index" in ts.features
                    assert ts.features["time_index"].numeric_compatible
                    assert "time_index" in ts.df
                assert all(data.temporal_covariates[0].df["time_index"] == expected_time_index_sample_0)
                assert all(data.temporal_covariates[1].df["time_index"] == expected_time_index_sample_1)
            else:
                assert "time_index" not in data.temporal_covariates.features

            if params["add_time_delta"] is True:
                assert "time_delta" in data.temporal_covariates.features
                assert data.temporal_covariates.features["time_delta"].numeric_compatible
                for ts in data.temporal_covariates:
                    assert "time_delta" in ts.features
                    assert ts.features["time_delta"].numeric_compatible
                    assert "time_delta" in ts.df
                assert all(data.temporal_covariates[0].df["time_delta"] == expected_time_delta_sample_0)
                assert all(data.temporal_covariates[1].df["time_delta"] == expected_time_delta_sample_1)
            else:
                assert "time_delta" not in data.temporal_covariates.features

        def test_raises_exception_if_neither_feature_specified(self):
            with pytest.raises(ValueError) as excinfo:
                TimeIndexFeatureConcatenator(params=dict(add_time_index=False, add_time_delta=False))
            assert "at least one of" in str(excinfo.value)

    class TestAddStaticCovariatesTC:
        @pytest.mark.parametrize(
            "t_cov_dfs, s_cov_df, params, expected_new_t_cov_dfs",
            [
                # Test case: default params.
                (
                    # t_cov_dfs:
                    [
                        pd.DataFrame({"a": [1, 2], "b": [1.1, 2.1]}, index=[1, 2]),
                        pd.DataFrame({"a": [7, 8, 9], "b": [7.1, 8.1, 9.1]}, index=[10, 30, 35]),
                    ],
                    # s_cov_df:
                    pd.DataFrame({"c": [11, 22], "d": [100.0, 200.0]}),
                    # params:
                    dict(
                        feature_name_prefix="static",
                        append_at_beginning=False,
                        drop_static_covariates=False,
                    ),
                    # expected_new_t_cov_dfs:
                    [
                        pd.DataFrame(
                            {"a": [1, 2], "b": [1.1, 2.1], "static_c": [11, 11], "static_d": [100.0, 100.0]},
                            index=[1, 2],
                        ),
                        pd.DataFrame(
                            {
                                "a": [7, 8, 9],
                                "b": [7.1, 8.1, 9.1],
                                "static_c": [22, 22, 22],
                                "static_d": [200.0, 200.0, 200.0],
                            },
                            index=[10, 30, 35],
                        ),
                    ],
                ),
                # Test case: default params, singular features.
                (
                    # t_cov_dfs:
                    [
                        pd.DataFrame({"a": [1, 2]}, index=[1, 2]),
                        pd.DataFrame({"a": [7, 8, 9]}, index=[10, 30, 35]),
                    ],
                    # s_cov_df:
                    pd.DataFrame({"c": [11, 22]}),
                    # params:
                    dict(
                        feature_name_prefix="static",
                        append_at_beginning=False,
                        drop_static_covariates=False,
                    ),
                    # expected_new_t_cov_dfs:
                    [
                        pd.DataFrame(
                            {"a": [1, 2], "static_c": [11, 11]},
                            index=[1, 2],
                        ),
                        pd.DataFrame(
                            {
                                "a": [7, 8, 9],
                                "static_c": [22, 22, 22],
                            },
                            index=[10, 30, 35],
                        ),
                    ],
                ),
                # Test case: append_at_beginning = True.
                (
                    # t_cov_dfs:
                    [
                        pd.DataFrame({"a": [1, 2], "b": [1.1, 2.1]}, index=[1, 2]),
                        pd.DataFrame({"a": [7, 8, 9], "b": [7.1, 8.1, 9.1]}, index=[10, 30, 35]),
                    ],
                    # s_cov_df:
                    pd.DataFrame({"c": [11, 22], "d": [100.0, 200.0]}),
                    # params:
                    dict(
                        feature_name_prefix="static",
                        append_at_beginning=True,
                        drop_static_covariates=False,
                    ),
                    # expected_new_t_cov_dfs:
                    [
                        pd.DataFrame(
                            {"static_c": [11, 11], "static_d": [100.0, 100.0], "a": [1, 2], "b": [1.1, 2.1]},
                            index=[1, 2],
                        ),
                        pd.DataFrame(
                            {
                                "static_c": [22, 22, 22],
                                "static_d": [200.0, 200.0, 200.0],
                                "a": [7, 8, 9],
                                "b": [7.1, 8.1, 9.1],
                            },
                            index=[10, 30, 35],
                        ),
                    ],
                ),
                # Test case: drop_static_covariates = True.
                (
                    # t_cov_dfs:
                    [
                        pd.DataFrame({"a": [1, 2], "b": [1.1, 2.1]}, index=[1, 2]),
                        pd.DataFrame({"a": [7, 8, 9], "b": [7.1, 8.1, 9.1]}, index=[10, 30, 35]),
                    ],
                    # s_cov_df:
                    pd.DataFrame({"c": [11, 22], "d": [100.0, 200.0]}),
                    # params:
                    dict(
                        feature_name_prefix="static",
                        append_at_beginning=False,
                        drop_static_covariates=True,
                    ),
                    # expected_new_t_cov_dfs:
                    [
                        pd.DataFrame(
                            {"a": [1, 2], "b": [1.1, 2.1], "static_c": [11, 11], "static_d": [100.0, 100.0]},
                            index=[1, 2],
                        ),
                        pd.DataFrame(
                            {
                                "a": [7, 8, 9],
                                "b": [7.1, 8.1, 9.1],
                                "static_c": [22, 22, 22],
                                "static_d": [200.0, 200.0, 200.0],
                            },
                            index=[10, 30, 35],
                        ),
                    ],
                ),
                # Test case: modify feature_name_prefix.
                (
                    # t_cov_dfs:
                    [
                        pd.DataFrame({"a": [1, 2], "b": [1.1, 2.1]}, index=[1, 2]),
                        pd.DataFrame({"a": [7, 8, 9], "b": [7.1, 8.1, 9.1]}, index=[10, 30, 35]),
                    ],
                    # s_cov_df:
                    pd.DataFrame({"c": [11, 22], "d": [100.0, 200.0]}),
                    # params:
                    dict(
                        feature_name_prefix="s",
                        append_at_beginning=False,
                        drop_static_covariates=False,
                    ),
                    # expected_new_t_cov_dfs:
                    [
                        pd.DataFrame(
                            {"a": [1, 2], "b": [1.1, 2.1], "s_c": [11, 11], "s_d": [100.0, 100.0]},
                            index=[1, 2],
                        ),
                        pd.DataFrame(
                            {
                                "a": [7, 8, 9],
                                "b": [7.1, 8.1, 9.1],
                                "s_c": [22, 22, 22],
                                "s_d": [200.0, 200.0, 200.0],
                            },
                            index=[10, 30, 35],
                        ),
                    ],
                ),
                # Test case: Set feature_name_prefix to None.
                (
                    # t_cov_dfs:
                    [
                        pd.DataFrame({"a": [1, 2], "b": [1.1, 2.1]}, index=[1, 2]),
                        pd.DataFrame({"a": [7, 8, 9], "b": [7.1, 8.1, 9.1]}, index=[10, 30, 35]),
                    ],
                    # s_cov_df:
                    pd.DataFrame({"c": [11, 22], "d": [100.0, 200.0]}),
                    # params:
                    dict(
                        feature_name_prefix=None,
                        append_at_beginning=False,
                        drop_static_covariates=False,
                    ),
                    # expected_new_t_cov_dfs:
                    [
                        pd.DataFrame(
                            {"a": [1, 2], "b": [1.1, 2.1], "c": [11, 11], "d": [100.0, 100.0]},
                            index=[1, 2],
                        ),
                        pd.DataFrame(
                            {
                                "a": [7, 8, 9],
                                "b": [7.1, 8.1, 9.1],
                                "c": [22, 22, 22],
                                "d": [200.0, 200.0, 200.0],
                            },
                            index=[10, 30, 35],
                        ),
                    ],
                ),
            ],
        )
        def test_successful_transform_numeric_features(self, t_cov_dfs, s_cov_df, params, expected_new_t_cov_dfs):
            # Arrange:
            data = Dataset(
                temporal_covariates=TimeSeriesSamples(t_cov_dfs),
                static_covariates=StaticSamples(s_cov_df),
            )
            transformer = StaticFeaturesConcatenator(params=params)

            # Act:
            data_new = transformer.fit_transform(data)

            # Assert:
            # - Check feature presence.
            for f in data.static_covariates.df.columns:
                assert (
                    transformer.params.feature_name_prefix + f"_{f}"
                    if transformer.params.feature_name_prefix is not None
                    else f
                ) in data_new.temporal_covariates.feature_names
            # - Check .static_covariates remains or not.
            if transformer.params.drop_static_covariates is True:
                assert data_new.static_covariates is None
            else:
                assert data_new.static_covariates is not None
                assert (data_new.static_covariates.df == s_cov_df).all().all()
            # - Check new temporal_covariates exactly.
            for sample_idx in [0, 1]:
                assert (data_new.temporal_covariates[sample_idx].df == expected_new_t_cov_dfs[sample_idx]).all().all()
            # - Check .features.
            assert len(data_new.temporal_covariates.features) == len(data.static_covariates.features) + len(
                data.temporal_covariates.features
            )
            assert all(
                [f in data_new.temporal_covariates.feature_names for f in data.temporal_covariates.feature_names]
            )
            assert all(
                [
                    (
                        transformer.params.feature_name_prefix + f"_{f}"
                        if transformer.params.feature_name_prefix is not None
                        else f
                    )
                    in data_new.temporal_covariates.feature_names
                    for f in data.static_covariates.feature_names
                ]
            )

        def test_successful_transform_numeric_categorical_features(self):
            # Arrange:
            t_cov_dfs = [
                pd.DataFrame({"a": [1, 2], "b": ["b1", "b2"]}, index=[1, 2]),
                pd.DataFrame({"a": [7, 8, 9], "b": ["b2", "b1", "b1"]}, index=[10, 30, 35]),
            ]
            s_cov_df = pd.DataFrame({"c": [11, 22], "d": ["d1", "d2"]})
            params = dict(
                feature_name_prefix="static",
                append_at_beginning=False,
                drop_static_covariates=False,
            )
            data = Dataset(
                temporal_covariates=TimeSeriesSamples(t_cov_dfs),
                static_covariates=StaticSamples(s_cov_df),
            )
            transformer = StaticFeaturesConcatenator(params=params)

            # Act:
            data_new = transformer.fit_transform(data)

            # Assert:
            expected_new_t_cov_dfs = [
                pd.DataFrame(
                    {"a": [1, 2], "b": ["b1", "b2"], "static_c": [11, 11], "static_d": ["d1", "d1"]},
                    index=[1, 2],
                ),
                pd.DataFrame(
                    {
                        "a": [7, 8, 9],
                        "b": ["b2", "b1", "b1"],
                        "static_c": [22, 22, 22],
                        "static_d": ["d2", "d2", "d2"],
                    },
                    index=[10, 30, 35],
                ),
            ]
            for sample_idx in [0, 1]:
                assert (data_new.temporal_covariates[sample_idx].df == expected_new_t_cov_dfs[sample_idx]).all().all()
            assert data_new.temporal_covariates.features["b"].categorical_compatible
            assert data_new.temporal_covariates.features["static_d"].categorical_compatible

        def test_raise_value_clashing_feature_names(self):
            # Arrange:
            t_cov_dfs = [
                pd.DataFrame({"a": [1, 2], "b": [1.1, 2.1]}, index=[1, 2]),
                pd.DataFrame({"a": [7, 8, 9], "b": [7.1, 8.1, 9.1]}, index=[10, 30, 35]),
            ]
            s_cov_df = pd.DataFrame({"a": [11, 22]})
            params = dict(
                feature_name_prefix=None,
                append_at_beginning=False,
                drop_static_covariates=False,
            )
            data = Dataset(
                temporal_covariates=TimeSeriesSamples(t_cov_dfs),
                static_covariates=StaticSamples(s_cov_df),
            )
            transformer = StaticFeaturesConcatenator(params=params)

            # Act, Assert:
            with pytest.raises(ValueError) as excinfo:
                transformer.fit_transform(data)
            assert "clash" in str(excinfo.value).lower()
