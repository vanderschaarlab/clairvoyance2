from unittest.mock import Mock, call

import numpy as np
import pandas as pd
import pytest

from clairvoyance2.data import Dataset, StaticSamples, TimeSeriesSamples

# pylint: disable=redefined-outer-name
# ^ Otherwise pylint trips up on pytest fixtures.

# pylint: disable=unused-argument
# ^ Some fixtures intentionally included without being explicitly used in test functions.


@pytest.fixture
def mocked_containers_for_repr():
    t_cov = Mock(spec=TimeSeriesSamples, features=("a", "b", "c"), n_samples=2)
    t_targ = Mock(spec=TimeSeriesSamples, features=("l"), n_samples=2)
    t_treat = Mock(spec=TimeSeriesSamples, features=("t"), n_samples=2)
    s_cov = Mock(spec=StaticSamples, features=("x", "y", "z"), n_samples=2)
    return t_cov, s_cov, t_targ, t_treat


@pytest.fixture
def mocked_containers_for_init():
    t_cov = Mock(TimeSeriesSamples)
    t_cov.sample_indices = list(range(10))
    t_cov.n_samples = 10
    t_cov.features = ("a", "b", "c")
    t_cov.name = "temporal_covariates"  # For tests only.
    t_targ = Mock(TimeSeriesSamples)
    t_targ.sample_indices = list(range(10))
    t_targ.n_samples = 10
    t_targ.features = "l"
    t_targ.name = "temporal_targets"  # For tests only.
    t_treat = Mock(TimeSeriesSamples)
    t_treat.sample_indices = list(range(10))
    t_treat.n_samples = 10
    t_treat.features = "t"
    t_treat.name = "temporal_treatments"  # For tests only.
    s_cov = Mock(StaticSamples)
    s_cov.sample_indices = list(range(10))
    s_cov.features = ("x", "y")
    s_cov.n_samples = 10
    s_cov.name = "static_covariates"  # For tests only.
    return t_cov, s_cov, t_targ, t_treat


@pytest.fixture
def patch_validate(monkeypatch):
    monkeypatch.setattr(
        "clairvoyance2.data.Dataset.validate",
        Mock(),
        raising=True,
    )


class TestRepr:
    def test_t_cov_s_cov_t_targ_t_treat(self, mocked_containers_for_repr, patch_validate):
        t_cov, s_cov, t_targ, t_treat = mocked_containers_for_repr

        ds = Dataset(
            temporal_covariates=t_cov, static_covariates=s_cov, temporal_targets=t_targ, temporal_treatments=t_treat
        )

        repr_ = str(ds)

        assert repr_.replace("    ", "").replace("\n", "") == (
            "Dataset(temporal_covariates=TimeSeriesSamples([2,*,3]),"
            "static_covariates=StaticSamples([2,3]),"
            "temporal_targets=TimeSeriesSamples([2,*,1]),"
            "temporal_treatments=TimeSeriesSamples([2,*,1]),)"
        )

    def test_t_cov_s_cov_t_targ(self, mocked_containers_for_repr, patch_validate):
        t_cov, s_cov, t_targ, _ = mocked_containers_for_repr

        ds = Dataset(temporal_covariates=t_cov, static_covariates=s_cov, temporal_targets=t_targ)

        repr_ = str(ds)

        assert repr_.replace("    ", "").replace("\n", "") == (
            "Dataset(temporal_covariates=TimeSeriesSamples([2,*,3]),"
            "static_covariates=StaticSamples([2,3]),"
            "temporal_targets=TimeSeriesSamples([2,*,1]),)"
        )

    def test_t_cov_t_targ(self, mocked_containers_for_repr, patch_validate):
        t_cov, _, t_targ, _ = mocked_containers_for_repr

        ds = Dataset(temporal_covariates=t_cov, temporal_targets=t_targ)

        repr_ = repr(ds)

        assert repr_.replace("    ", "").replace("\n", "") == (
            "Dataset(temporal_covariates=TimeSeriesSamples([2,*,3]),temporal_targets=TimeSeriesSamples([2,*,1]),)"
        )

    def test_t_cov_s_cov(self, mocked_containers_for_repr, patch_validate):
        t_cov, s_cov, _, _ = mocked_containers_for_repr

        ds = Dataset(temporal_covariates=t_cov, static_covariates=s_cov)

        repr_ = repr(ds)

        assert repr_.replace("    ", "").replace("\n", "") == (
            "Dataset(temporal_covariates=TimeSeriesSamples([2,*,3]),static_covariates=StaticSamples([2,3]),)"
        )

    def test_t_cov(self, mocked_containers_for_repr, patch_validate):
        t_cov, _, _, _ = mocked_containers_for_repr

        ds = Dataset(temporal_covariates=t_cov)

        repr_ = repr(ds)

        assert repr_.replace("    ", "").replace("\n", "") == (
            "Dataset(temporal_covariates=TimeSeriesSamples([2,*,3]),)"
        )


class TestInit:
    def test_success_t_cov(self, mocked_containers_for_init):
        t_cov, _, _, _ = mocked_containers_for_init
        Dataset(temporal_covariates=t_cov)

    def test_success_t_cov_t_targ(self, mocked_containers_for_init):
        t_cov, _, t_targ, _ = mocked_containers_for_init
        Dataset(temporal_covariates=t_cov, temporal_targets=t_targ)

    def test_success_t_cov_s_cov(self, mocked_containers_for_init):
        t_cov, s_cov, _, _ = mocked_containers_for_init
        Dataset(temporal_covariates=t_cov, static_covariates=s_cov)

    def test_success_t_cov_s_cov_t_targ(self, mocked_containers_for_init):
        t_cov, s_cov, t_targ, _ = mocked_containers_for_init
        Dataset(temporal_covariates=t_cov, static_covariates=s_cov, temporal_targets=t_targ)

    def test_success_t_cov_s_cov_t_targ_t_treat(self, mocked_containers_for_init):
        t_cov, s_cov, t_targ, t_treat = mocked_containers_for_init
        Dataset(
            temporal_covariates=t_cov, static_covariates=s_cov, temporal_targets=t_targ, temporal_treatments=t_treat
        )

    def test_fail_s_cov_wrong_n_samples(self, mocked_containers_for_init):
        t_cov, s_cov, _, _ = mocked_containers_for_init
        s_cov.n_samples = 999
        with pytest.raises(ValueError) as excinfo:
            Dataset(temporal_covariates=t_cov, static_covariates=s_cov)
        assert "samples" in str(excinfo.value) and "static_covariates" in str(excinfo.value)

    def test_fail_t_targ_wrong_n_samples(self, mocked_containers_for_init):
        t_cov, _, t_targ, _ = mocked_containers_for_init
        t_targ.n_samples = 999
        with pytest.raises(ValueError) as excinfo:
            Dataset(temporal_covariates=t_cov, temporal_targets=t_targ)
        assert "samples" in str(excinfo.value) and "temporal_targets" in str(excinfo.value)

    def test_fail_t_treat_wrong_n_samples(self, mocked_containers_for_init):
        t_cov, _, t_targ, t_treat = mocked_containers_for_init
        t_treat.n_samples = 999
        with pytest.raises(ValueError) as excinfo:
            Dataset(temporal_covariates=t_cov, temporal_targets=t_targ, temporal_treatments=t_treat)
        assert "samples" in str(excinfo.value) and "temporal_treatments" in str(excinfo.value)

    @pytest.mark.parametrize("sample_indices", [list(range(0, -10, -1)), [1, 2, 3]])
    def test_fail_s_cov_wrong_sample_indices(self, sample_indices, mocked_containers_for_init):
        t_cov, s_cov, _, _ = mocked_containers_for_init
        s_cov.sample_indices = sample_indices
        with pytest.raises(ValueError) as excinfo:
            Dataset(temporal_covariates=t_cov, static_covariates=s_cov)
        assert "sample indices" in str(excinfo.value) and "static_covariates" in str(excinfo.value)

    def test_fail_t_targ_wrong_sample_indices(self, mocked_containers_for_init):
        t_cov, _, t_targ, _ = mocked_containers_for_init
        t_targ.sample_indices = list(range(0, -10, -1))
        with pytest.raises(ValueError) as excinfo:
            Dataset(temporal_covariates=t_cov, temporal_targets=t_targ)
        assert "sample indices" in str(excinfo.value) and "temporal_targets" in str(excinfo.value)

    def test_fail_t_treat_wrong_sample_indices(self, mocked_containers_for_init):
        t_cov, _, t_targ, t_treat = mocked_containers_for_init
        t_treat.sample_indices = list(range(0, -10, -1))
        with pytest.raises(ValueError) as excinfo:
            Dataset(temporal_covariates=t_cov, temporal_targets=t_targ, temporal_treatments=t_treat)
        assert "sample indices" in str(excinfo.value) and "temporal_treatments" in str(excinfo.value)

    class TestInitInnerContainers:
        def test_init_static_covariates(self, mocked_containers_for_init, monkeypatch):
            (t_cov, *_) = mocked_containers_for_init
            static_samples_init = Mock(return_value=None)
            static_covariates_data = Mock()
            mock_missing_indicator = Mock()
            monkeypatch.setattr(
                "clairvoyance2.data.dataformat.StaticSamples.__init__",
                static_samples_init,
                raising=True,
            )
            monkeypatch.setattr(
                "clairvoyance2.data.dataset.Dataset.validate",
                Mock(),
                raising=True,
            )

            ds = Dataset(
                temporal_covariates=t_cov,
                static_covariates=static_covariates_data,
                missing_indicator=mock_missing_indicator,
            )

            assert id(ds.temporal_covariates) == id(t_cov)
            static_samples_init.assert_called_once_with(
                static_covariates_data,
                sample_indices=None,
                categorical_features=tuple(),
                missing_indicator=mock_missing_indicator,
            )

        def test_init_temporal_targets(self, mocked_containers_for_init, monkeypatch):
            (t_cov, *_) = mocked_containers_for_init
            time_series_samples_init = Mock(return_value=None)
            temporal_targets_data = Mock()
            mock_missing_indicator = Mock()
            monkeypatch.setattr(
                "clairvoyance2.data.dataformat.TimeSeriesSamples.__init__",
                time_series_samples_init,
                raising=True,
            )
            monkeypatch.setattr(
                "clairvoyance2.data.dataset.Dataset.validate",
                Mock(),
                raising=True,
            )

            ds = Dataset(
                temporal_covariates=t_cov,
                temporal_targets=temporal_targets_data,
                missing_indicator=mock_missing_indicator,
            )

            assert id(ds.temporal_covariates) == id(t_cov)
            time_series_samples_init.assert_called_once_with(
                temporal_targets_data,
                sample_indices=None,
                categorical_features=tuple(),
                missing_indicator=mock_missing_indicator,
            )

        def test_init_many(self, mocked_containers_for_init, monkeypatch):
            (t_cov, *_) = mocked_containers_for_init
            static_samples_init = Mock(return_value=None)
            time_series_samples_init = Mock(return_value=None)
            static_covariates_data = Mock()
            temporal_targets_data = Mock()
            temporal_treatments_data = Mock()
            mock_missing_indicator = Mock()
            monkeypatch.setattr(
                "clairvoyance2.data.dataformat.StaticSamples.__init__",
                static_samples_init,
                raising=True,
            )
            monkeypatch.setattr(
                "clairvoyance2.data.dataformat.TimeSeriesSamples.__init__",
                time_series_samples_init,
                raising=True,
            )
            monkeypatch.setattr(
                "clairvoyance2.data.dataset.Dataset.validate",
                Mock(),
                raising=True,
            )

            ds = Dataset(
                temporal_covariates=t_cov,
                static_covariates=static_covariates_data,
                temporal_targets=temporal_targets_data,
                temporal_treatments=temporal_treatments_data,
                missing_indicator=mock_missing_indicator,
            )

            assert id(ds.temporal_covariates) == id(t_cov)
            static_samples_init.assert_has_calls(
                [
                    call(
                        static_covariates_data,
                        sample_indices=None,
                        categorical_features=tuple(),
                        missing_indicator=mock_missing_indicator,
                    ),
                ]
            )
            time_series_samples_init.assert_has_calls(
                [
                    call(
                        temporal_targets_data,
                        sample_indices=None,
                        categorical_features=tuple(),
                        missing_indicator=mock_missing_indicator,
                    ),
                    call(
                        temporal_treatments_data,
                        sample_indices=None,
                        categorical_features=tuple(),
                        missing_indicator=mock_missing_indicator,
                    ),
                ]
            )

        def test_init_many_with_categorical(self, mocked_containers_for_init, monkeypatch):
            (t_cov, *_) = mocked_containers_for_init
            static_samples_init = Mock(return_value=None)
            time_series_samples_init = Mock(return_value=None)
            static_covariates_data = Mock()
            temporal_targets_data = Mock()
            temporal_treatments_data = Mock()
            mock_missing_indicator = Mock()
            categorical_features = dict(
                temporal_covariates="c_t_cov",
                static_covariates="c_s_cov",
                temporal_targets="c_t_targ",
                temporal_treatments="c_t_treat",
            )
            monkeypatch.setattr(
                "clairvoyance2.data.dataformat.StaticSamples.__init__",
                static_samples_init,
                raising=True,
            )
            monkeypatch.setattr(
                "clairvoyance2.data.dataformat.TimeSeriesSamples.__init__",
                time_series_samples_init,
                raising=True,
            )
            monkeypatch.setattr(
                "clairvoyance2.data.dataset.Dataset.validate",
                Mock(),
                raising=True,
            )

            ds = Dataset(
                temporal_covariates=t_cov,
                static_covariates=static_covariates_data,
                temporal_targets=temporal_targets_data,
                temporal_treatments=temporal_treatments_data,
                sample_indices=None,
                categorical_features=categorical_features,
                missing_indicator=mock_missing_indicator,
            )

            assert id(ds.temporal_covariates) == id(t_cov)
            static_samples_init.assert_has_calls(
                [
                    call(
                        static_covariates_data,
                        sample_indices=None,
                        categorical_features="c_s_cov",
                        missing_indicator=mock_missing_indicator,
                    ),
                ]
            )
            time_series_samples_init.assert_has_calls(
                [
                    call(
                        temporal_targets_data,
                        sample_indices=None,
                        categorical_features="c_t_targ",
                        missing_indicator=mock_missing_indicator,
                    ),
                    call(
                        temporal_treatments_data,
                        sample_indices=None,
                        categorical_features="c_t_treat",
                        missing_indicator=mock_missing_indicator,
                    ),
                ]
            )

        def test_init_many_with_sample_indices(self, mocked_containers_for_init, monkeypatch):
            (t_cov, *_) = mocked_containers_for_init
            static_samples_init = Mock(return_value=None)
            time_series_samples_init = Mock(return_value=None)
            static_covariates_data = Mock()
            temporal_targets_data = Mock()
            temporal_treatments_data = Mock()
            mock_missing_indicator = Mock()
            sample_indices = Mock()
            monkeypatch.setattr(
                "clairvoyance2.data.dataformat.StaticSamples.__init__",
                static_samples_init,
                raising=True,
            )
            monkeypatch.setattr(
                "clairvoyance2.data.dataformat.TimeSeriesSamples.__init__",
                time_series_samples_init,
                raising=True,
            )
            monkeypatch.setattr(
                "clairvoyance2.data.dataset.Dataset.validate",
                Mock(),
                raising=True,
            )

            ds = Dataset(
                temporal_covariates=t_cov,
                static_covariates=static_covariates_data,
                temporal_targets=temporal_targets_data,
                temporal_treatments=temporal_treatments_data,
                sample_indices=sample_indices,
                missing_indicator=mock_missing_indicator,
            )

            assert id(ds.temporal_covariates) == id(t_cov)
            static_samples_init.assert_has_calls(
                [
                    call(
                        static_covariates_data,
                        sample_indices=sample_indices,
                        categorical_features=tuple(),
                        missing_indicator=mock_missing_indicator,
                    ),
                ]
            )
            time_series_samples_init.assert_has_calls(
                [
                    call(
                        temporal_targets_data,
                        sample_indices=sample_indices,
                        categorical_features=tuple(),
                        missing_indicator=mock_missing_indicator,
                    ),
                    call(
                        temporal_treatments_data,
                        sample_indices=sample_indices,
                        categorical_features=tuple(),
                        missing_indicator=mock_missing_indicator,
                    ),
                ]
            )

    @pytest.mark.parametrize(
        "categorical_features",
        [
            dict(random="something"),
            dict(
                temporal_covariates="something",
                random="something",
                temporal_targets="something",
                temporal_treatments="something",
            ),
            dict(
                temporal_covariates="something",
                static_covariates="something",
                temporal_targets="something",
                temporal_treatments="something",
                random="something",
            ),
        ],
    )
    def test_categorical_features_arg_fails_wrong_container_name(
        self, categorical_features, mocked_containers_for_init
    ):
        t_cov, s_cov, t_targ, t_treat = mocked_containers_for_init

        with pytest.raises(ValueError) as excinfo:
            Dataset(
                temporal_covariates=t_cov,
                static_covariates=s_cov,
                temporal_targets=t_targ,
                temporal_treatments=t_treat,
                categorical_features=categorical_features,
            )
        assert "unexpected key" in str(excinfo.value).lower() and "categorical_features" in str(excinfo.value)


def test_n_samples(mocked_containers_for_init):
    (t_cov, *_) = mocked_containers_for_init
    ds = Dataset(temporal_covariates=t_cov)

    n_samples = ds.n_samples

    assert n_samples == 10


def test_sample_indices(mocked_containers_for_init):
    (t_cov, *_) = mocked_containers_for_init
    ds = Dataset(temporal_covariates=t_cov)

    sample_indices = ds.sample_indices

    assert sample_indices == list(range(10))


class TestDataContainerHelpers:
    def test_static_data_containers(self, mocked_containers_for_init):
        t_cov, s_cov, t_targ, t_treat = mocked_containers_for_init
        ds = Dataset(
            temporal_covariates=t_cov, static_covariates=s_cov, temporal_targets=t_targ, temporal_treatments=t_treat
        )

        static_data_containers = ds.static_data_containers

        assert isinstance(static_data_containers, dict)
        assert len(static_data_containers) == 1
        assert "static_covariates" == list(static_data_containers.keys())[0]
        for name, value in static_data_containers.items():
            assert name == value.name

    def test_temporal_data_containers(self, mocked_containers_for_init):
        t_cov, s_cov, t_targ, t_treat = mocked_containers_for_init
        ds = Dataset(
            temporal_covariates=t_cov, static_covariates=s_cov, temporal_targets=t_targ, temporal_treatments=t_treat
        )

        temporal_data_containers = ds.temporal_data_containers

        assert isinstance(temporal_data_containers, dict)
        assert len(temporal_data_containers) == 3
        assert "temporal_covariates" == list(temporal_data_containers.keys())[0]
        assert "temporal_targets" == list(temporal_data_containers.keys())[1]
        assert "temporal_treatments" == list(temporal_data_containers.keys())[2]
        for name, value in temporal_data_containers.items():
            assert name == value.name

    def test_all_data_containers(self, mocked_containers_for_init):
        t_cov, s_cov, t_targ, t_treat = mocked_containers_for_init
        ds = Dataset(
            temporal_covariates=t_cov, static_covariates=s_cov, temporal_targets=t_targ, temporal_treatments=t_treat
        )

        all_data_containers = ds.all_data_containers

        assert isinstance(all_data_containers, dict)
        assert len(all_data_containers) == 4
        assert "temporal_covariates" == list(all_data_containers.keys())[0]
        assert "static_covariates" == list(all_data_containers.keys())[1]
        assert "temporal_targets" == list(all_data_containers.keys())[2]
        assert "temporal_treatments" == list(all_data_containers.keys())[3]
        for name, value in all_data_containers.items():
            assert name == value.name

    def test_data_containers_added_later(self, mocked_containers_for_init):
        # Arrange:
        t_cov, s_cov, t_targ, t_treat = mocked_containers_for_init
        ds = Dataset(temporal_covariates=t_cov)

        # Act:
        ds.static_covariates = s_cov
        ds.temporal_targets = t_targ
        ds.temporal_treatments = t_treat
        static_data_containers = ds.static_data_containers
        temporal_data_containers = ds.temporal_data_containers
        all_data_containers = ds.all_data_containers

        # Assert:
        assert isinstance(static_data_containers, dict)
        assert len(static_data_containers) == 1
        assert "static_covariates" == list(static_data_containers.keys())[0]
        for name, value in static_data_containers.items():
            assert name == value.name

        assert isinstance(temporal_data_containers, dict)
        assert len(temporal_data_containers) == 3
        assert "temporal_covariates" == list(temporal_data_containers.keys())[0]
        assert "temporal_targets" == list(temporal_data_containers.keys())[1]
        assert "temporal_treatments" == list(temporal_data_containers.keys())[2]
        for name, value in temporal_data_containers.items():
            assert name == value.name

        assert isinstance(all_data_containers, dict)
        assert len(all_data_containers) == 4
        assert "temporal_covariates" == list(all_data_containers.keys())[0]
        assert "static_covariates" == list(all_data_containers.keys())[1]
        assert "temporal_targets" == list(all_data_containers.keys())[2]
        assert "temporal_treatments" == list(all_data_containers.keys())[3]
        for name, value in all_data_containers.items():
            assert name == value.name


class TestSequenceAPI:
    def test_py_sequence_some_methods_not_implemented(self, monkeypatch):
        monkeypatch.setattr(
            "clairvoyance2.data.dataset.Dataset.validate",
            Mock(),
            raising=True,
        )
        monkeypatch.setattr(
            "clairvoyance2.data.dataset.Dataset.__init__",
            Mock(return_value=None),
            raising=True,
        )
        ts = Dataset(temporal_covariates=Mock())

        with pytest.raises(NotImplementedError):
            ts.count(None)

        with pytest.raises(NotImplementedError):
            ts.index(None)


@pytest.fixture
def containers_for_dataset_init():
    sample_indices = [1, 3, 5]
    t_cov = TimeSeriesSamples(
        [
            pd.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0], "c": [0, 0, 1]}),
            pd.DataFrame({"a": [7, 8], "b": [7.0, 8.0], "c": [0, 0]}),
            pd.DataFrame({"a": [1, 1, 2, 2, 4], "b": [1.0, 5.0, 5.0, 7.0, 9.0], "c": [0, 0, 1, 1, 0]}),
        ],
        sample_indices=sample_indices,
    )
    s_cov = StaticSamples(
        pd.DataFrame({"s_a": [0, 1, 0], "s_b": [33.0, 32.0, 21.0]}, index=sample_indices), categorical_features=["s_a"]
    )
    t_targ = TimeSeriesSamples(
        [
            pd.DataFrame({"t": [1, 1, 0], "o": [1.1, 7.1, 3.1]}),
            pd.DataFrame({"t": [0, 0], "o": [7.3, 8.3]}),
            pd.DataFrame({"t": [1, 1, 0, 0, 0], "o": [1.2, 5.3, 5.1, 7.1, 9.1]}),
        ],
        sample_indices=sample_indices,
        categorical_features=["t"],
    )
    t_treat = TimeSeriesSamples(
        [
            pd.DataFrame({"q": [1.4, 1.4, 0.4]}),
            pd.DataFrame({"q": [0.4, 0.4]}),
            pd.DataFrame({"q": [1.4, 1.4, 0.4, 0.4, 0.4]}),
        ],
        sample_indices=sample_indices,
    )
    return t_cov, s_cov, t_targ, t_treat


class TestIntegration:
    class TestInit:
        @pytest.mark.parametrize("sample_indices, sample_indices_expected", [(None, [0, 1]), ([4, 8], [4, 8])])
        def test_success_mixed_initialization_of_inner_containers(self, sample_indices, sample_indices_expected):
            # Arrange.
            t_cov = TimeSeriesSamples(
                [
                    pd.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0], "c": [0, 0, 1]}),
                    pd.DataFrame({"a": [7, 8, 9], "b": [7.0, 8.0, 9.0], "c": [0, 0, 0]}),
                ],
                sample_indices=sample_indices,
                missing_indicator=np.nan,
            )
            s_cov = np.asarray(
                [
                    [1.1, 11.0, 7.4],
                    [0.3, 12.0, 3.3],
                ],
            )
            t_targ = [
                pd.DataFrame({"t": [1, 2, 3], "o": [0, 0, 1]}),
                pd.DataFrame({"t": [7, 8, 9], "o": [0, 0, 1]}),
            ]
            t_treat = np.asarray(
                [
                    np.asarray([[0, 1, 1], [11, 12, 13], [4, 5, 2], [3, 7, 2]]).T,
                    np.asarray([[0, 0, 0], [-11, -12, -13], [4, 5, 2], [2, 8, 4]]).T,
                ],
                dtype=int,
            )
            categorical_features = dict(temporal_targets=["o"], temporal_treatments=[0, 3])

            # Act.
            ds = Dataset(
                temporal_covariates=t_cov,
                static_covariates=s_cov,
                temporal_targets=t_targ,
                temporal_treatments=t_treat,
                sample_indices=sample_indices,
                categorical_features=categorical_features,
                missing_indicator=np.nan,
            )

            # Assert.
            assert ds.temporal_covariates is not None
            assert ds.static_covariates is not None
            assert ds.temporal_targets is not None
            assert ds.temporal_treatments is not None
            # ---
            assert isinstance(ds.temporal_covariates, TimeSeriesSamples)
            assert isinstance(ds.static_covariates, StaticSamples)
            assert isinstance(ds.temporal_targets, TimeSeriesSamples)
            assert isinstance(ds.temporal_treatments, TimeSeriesSamples)
            # ---
            assert ds.n_samples == 2
            assert ds.sample_indices == sample_indices_expected
            # ---
            assert ds.temporal_covariates.n_samples == 2
            assert ds.static_covariates.n_samples == 2
            assert ds.temporal_targets.n_samples == 2
            assert ds.temporal_treatments.n_samples == 2
            # ---
            assert ds.temporal_covariates.sample_indices == sample_indices_expected
            assert ds.static_covariates.sample_indices == sample_indices_expected
            assert ds.temporal_targets.sample_indices == sample_indices_expected
            assert ds.temporal_treatments.sample_indices == sample_indices_expected
            # ---
            assert ds.temporal_covariates.n_timesteps_per_sample == [3, 3]
            assert ds.temporal_targets.n_timesteps_per_sample == [3, 3]
            assert ds.temporal_treatments.n_timesteps_per_sample == [3, 3]
            # ---
            assert ds.temporal_covariates.n_features == 3
            assert ds.static_covariates.n_features == 3
            assert ds.temporal_targets.n_features == 2
            assert ds.temporal_treatments.n_features == 4
            # ---
            assert "o" in ds.temporal_targets.categorical_def
            assert 0 in ds.temporal_treatments.categorical_def and 3 in ds.temporal_treatments.categorical_def

    class TestSequenceAPI:
        def test_getitem_success(self, containers_for_dataset_init):
            # Arrange:
            t_cov, s_cov, t_targ, t_treat = containers_for_dataset_init
            ds = Dataset(
                temporal_covariates=t_cov,
                static_covariates=s_cov,
                temporal_targets=t_targ,
                temporal_treatments=t_treat,
            )

            # Act:
            index = 3
            ds_item = ds[index]

            # Assert:
            assert len(ds_item) == 1
            assert ds_item.n_samples == 1
            assert ds_item.sample_indices == [index]
            assert ds_item.temporal_covariates is not None
            assert ds_item.static_covariates is not None
            assert ds_item.temporal_targets is not None
            assert ds_item.temporal_treatments is not None
            assert ds_item.temporal_covariates.n_features == 3
            assert ds_item.static_covariates.n_features == 2
            assert ds_item.temporal_targets.n_features == 2
            assert ds_item.temporal_treatments.n_features == 1
            assert "t" in ds_item.temporal_targets.categorical_def
            assert "s_a" in ds_item.static_covariates.categorical_def
            assert ds_item.temporal_covariates.sample_indices == [index]
            assert ds_item.static_covariates.sample_indices == [index]
            assert ds_item.temporal_targets.sample_indices == [index]
            assert ds_item.temporal_treatments.sample_indices == [index]
            assert (
                (ds_item.temporal_covariates[index].df == pd.DataFrame({"a": [7, 8], "b": [7.0, 8.0], "c": [0, 0]}))
                .all()
                .all()
            )
            assert (ds_item.temporal_targets[index].df == pd.DataFrame({"t": [0, 0], "o": [7.3, 8.3]})).all().all()
            assert (ds_item.temporal_treatments[index].df == pd.DataFrame({"q": [0.4, 0.4]})).all().all()
            # TODO: Note the below inconsistency in that the DF here has the sample index.
            assert (
                (ds_item.static_covariates[index].df == pd.DataFrame({"s_a": [1], "s_b": [32.0]}, index=[index]))
                .all()
                .all()
            )

        def test_getitem_raises_key_error(self, containers_for_dataset_init):
            t_cov, s_cov, t_targ, t_treat = containers_for_dataset_init
            ds = Dataset(
                temporal_covariates=t_cov,
                static_covariates=s_cov,
                temporal_targets=t_targ,
                temporal_treatments=t_treat,
            )

            with pytest.raises(KeyError) as excinfo:
                index = "wrong_key"
                _ = ds[index]
            assert "one of types" in str(excinfo.value) and "str" in str(excinfo.value)

        def test_iter(self, containers_for_dataset_init):
            # Arrange:
            t_cov, s_cov, t_targ, t_treat = containers_for_dataset_init
            ds = Dataset(
                temporal_covariates=t_cov,
                static_covariates=s_cov,
                temporal_targets=t_targ,
                temporal_treatments=t_treat,
            )

            # Act / Assert:
            for index, ds_item in zip([1, 3, 5], ds):
                assert len(ds_item) == 1
                assert ds_item.n_samples == 1
                assert ds_item.sample_indices == [index]
                assert ds_item.temporal_covariates is not None
                assert ds_item.static_covariates is not None
                assert ds_item.temporal_targets is not None
                assert ds_item.temporal_treatments is not None

        def test_reversed(self, containers_for_dataset_init):
            # Arrange:
            t_cov, s_cov, t_targ, t_treat = containers_for_dataset_init
            ds = Dataset(
                temporal_covariates=t_cov,
                static_covariates=s_cov,
                temporal_targets=t_targ,
                temporal_treatments=t_treat,
            )

            # Act / Assert:
            for index, ds_item in zip([5, 3, 1], reversed(ds)):
                assert len(ds_item) == 1
                assert ds_item.n_samples == 1
                assert ds_item.sample_indices == [index]
                assert ds_item.temporal_covariates is not None
                assert ds_item.static_covariates is not None
                assert ds_item.temporal_targets is not None
                assert ds_item.temporal_treatments is not None

        def test_contains(self, containers_for_dataset_init):
            # Arrange:
            t_cov, s_cov, t_targ, t_treat = containers_for_dataset_init
            ds = Dataset(
                temporal_covariates=t_cov,
                static_covariates=s_cov,
                temporal_targets=t_targ,
                temporal_treatments=t_treat,
            )

            # Act / Assert:
            assert 3 in ds
            assert not (9 in ds)
