from unittest.mock import Mock

# import pandas as pd
import pytest

from clairvoyance2.data import Dataset, StaticSamples, TimeSeriesSamples

# pylint: disable=redefined-outer-name
# ^ Otherwise pylint trips up on pytest fixtures.

# pylint: disable=unused-argument
# ^ Some fixtures intentionally included without being explicitly used in test functions.


@pytest.fixture
def mocked_containers_for_repr():
    t_cov = Mock(features=("a", "b", "c"), n_samples=2)
    t_cov.__class__.__name__ = "TimeSeriesSamples"
    t_targ = Mock(features=("l"), n_samples=2)
    t_targ.__class__.__name__ = "TimeSeriesSamples"
    s_cov = Mock(features=("x", "y", "z"), n_samples=2)
    s_cov.__class__.__name__ = "StaticSamples"
    return t_cov, s_cov, t_targ


@pytest.fixture
def mocked_containers_for_init():
    t_cov = Mock(TimeSeriesSamples)
    t_cov.n_samples = 10
    t_cov.features = ("a", "b", "c")
    t_targ = Mock(TimeSeriesSamples)
    t_targ.n_samples = 10
    t_targ.features = "l"
    s_cov = Mock(StaticSamples)
    s_cov.features = ("x", "y")
    s_cov.n_samples = 10
    return t_cov, s_cov, t_targ


@pytest.fixture
def patch_validate(monkeypatch):
    monkeypatch.setattr(
        "clairvoyance2.data.Dataset.validate",
        Mock(),
        raising=True,
    )


class TestRepr:
    def test_t_cov_s_cov_t_targ(self, mocked_containers_for_repr, patch_validate):
        t_cov, s_cov, t_targ = mocked_containers_for_repr

        ds = Dataset(temporal_covariates=t_cov, static_covariates=s_cov, temporal_targets=t_targ)

        repr_ = str(ds)

        assert repr_.replace("    ", "").replace("\n", "") == (
            "Dataset(temporal_covariates=TimeSeriesSamples([2,*,3]),"
            "static_covariates=StaticSamples([2,3]),"
            "temporal_targets=TimeSeriesSamples([2,*,1]),)"
        )

    def test_t_cov_t_targ(self, mocked_containers_for_repr, patch_validate):
        t_cov, _, t_targ = mocked_containers_for_repr

        ds = Dataset(temporal_covariates=t_cov, temporal_targets=t_targ)

        repr_ = repr(ds)

        assert repr_.replace("    ", "").replace("\n", "") == (
            "Dataset(temporal_covariates=TimeSeriesSamples([2,*,3]),temporal_targets=TimeSeriesSamples([2,*,1]),)"
        )

    def test_t_cov_s_cov(self, mocked_containers_for_repr, patch_validate):
        t_cov, s_cov, _ = mocked_containers_for_repr

        ds = Dataset(temporal_covariates=t_cov, static_covariates=s_cov)

        repr_ = repr(ds)

        assert repr_.replace("    ", "").replace("\n", "") == (
            "Dataset(temporal_covariates=TimeSeriesSamples([2,*,3]),static_covariates=StaticSamples([2,3]),)"
        )

    def test_t_cov(self, mocked_containers_for_repr, patch_validate):
        t_cov, _, _ = mocked_containers_for_repr

        ds = Dataset(temporal_covariates=t_cov)

        repr_ = repr(ds)

        assert repr_.replace("    ", "").replace("\n", "") == (
            "Dataset(temporal_covariates=TimeSeriesSamples([2,*,3]),)"
        )


class TestInit:
    def test_success_t_cov(self, mocked_containers_for_init):
        t_cov, _, _ = mocked_containers_for_init
        Dataset(temporal_covariates=t_cov)

    def test_success_t_cov_t_targ(self, mocked_containers_for_init):
        t_cov, _, t_targ = mocked_containers_for_init
        Dataset(temporal_covariates=t_cov, temporal_targets=t_targ)

    def test_success_t_cov_s_cov(self, mocked_containers_for_init):
        t_cov, s_cov, _ = mocked_containers_for_init
        Dataset(temporal_covariates=t_cov, static_covariates=s_cov)

    def test_success_t_cov_s_cov_t_targ(self, mocked_containers_for_init):
        t_cov, s_cov, t_targ = mocked_containers_for_init
        Dataset(temporal_covariates=t_cov, static_covariates=s_cov, temporal_targets=t_targ)

    def test_fail_t_cov_wrong_type(self):
        t_cov = Mock(StaticSamples)
        with pytest.raises(TypeError) as excinfo:
            Dataset(temporal_covariates=t_cov)
        assert "TimeSeriesSamples" in str(excinfo.value) and "temporal_covariates" in str(excinfo.value)

    def test_fail_s_cov_wrong_type(self, mocked_containers_for_init):
        t_cov, _, _ = mocked_containers_for_init
        s_cov = Mock(TimeSeriesSamples)
        with pytest.raises(TypeError) as excinfo:
            Dataset(temporal_covariates=t_cov, static_covariates=s_cov)
        assert "StaticSamples" in str(excinfo.value) and "static_covariates" in str(excinfo.value)

    def test_fail_t_targ_wrong_type(self, mocked_containers_for_init):
        t_cov, s_cov, _ = mocked_containers_for_init
        t_targ = Mock(StaticSamples)
        with pytest.raises(TypeError) as excinfo:
            Dataset(temporal_covariates=t_cov, static_covariates=s_cov, temporal_targets=t_targ)
        assert "TimeSeriesSamples" in str(excinfo.value) and "temporal_targets" in str(excinfo.value)

    def test_fail_s_cov_wrong_n_samples(self, mocked_containers_for_init):
        t_cov, s_cov, _ = mocked_containers_for_init
        s_cov.n_samples = 999
        with pytest.raises(ValueError) as excinfo:
            Dataset(temporal_covariates=t_cov, static_covariates=s_cov)
        assert "samples" in str(excinfo.value) and "static_covariates" in str(excinfo.value)

    def test_fail_t_targ_wrong_n_samples(self, mocked_containers_for_init):
        t_cov, _, t_targ = mocked_containers_for_init
        t_targ.n_samples = 999
        with pytest.raises(ValueError) as excinfo:
            Dataset(temporal_covariates=t_cov, temporal_targets=t_targ)
        assert "samples" in str(excinfo.value) and "temporal_targets" in str(excinfo.value)


def test_unpack(mocked_containers_for_init):
    t_cov, s_cov, t_targ = mocked_containers_for_init

    # For assert only:
    t_cov.name = "t_cov"
    s_cov.name = "s_cov"
    t_targ.name = "t_targ"

    ds = Dataset(temporal_covariates=t_cov, static_covariates=s_cov, temporal_targets=t_targ)

    t_cov, s_cov, t_targ = ds

    assert isinstance(t_cov, TimeSeriesSamples)
    assert isinstance(s_cov, StaticSamples)
    assert isinstance(t_targ, TimeSeriesSamples)
    assert t_cov.name == "t_cov"
    assert s_cov.name == "s_cov"
    assert t_targ.name == "t_targ"


def test_n_samples(mocked_containers_for_init):
    t_cov, _, _ = mocked_containers_for_init
    ds = Dataset(temporal_covariates=t_cov)

    n_samples = ds.n_samples

    assert n_samples == 10
