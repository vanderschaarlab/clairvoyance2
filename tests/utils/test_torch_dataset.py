from unittest.mock import Mock

import numpy as np
import pytest
import torch

from clairvoyance2.data import Dataset, StaticSamples, TimeSeriesSamples
from clairvoyance2.utils.torch_dataset import (
    ClairvoyanceTorchDataset,  # TODO: May remove this.
)
from clairvoyance2.utils.torch_dataset import CustomTorchDataset

# pylint: disable=redefined-outer-name
# ^ Otherwise pylint trips up on pytest fixtures.


@pytest.fixture
def a() -> np.ndarray:
    a_ = np.asarray([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    a_ = np.asarray([a_ * (idx + 1) for idx in range(10)])
    return a_


@pytest.fixture
def b(a) -> np.ndarray:
    return -a


class TestIntegration:
    class TestToClairvoyanceTorchDatasetFunction:
        def test_temporal_covariates_only(self):
            t_cov = TimeSeriesSamples(
                [
                    np.asarray([[1.0, 2.0, 3.0, 4.0], [11.0, 12.0, 13.0, 14.0], [111.0, 112.0, 113.0, 114.0]]).T,
                    np.asarray(
                        [[-1.0, -2.0, -3.0, -4.0], [-11.0, -12.0, -13.0, -14.0], [-111.0, -112.0, -113.0, -114.0]]
                    ).T,
                ]
            )
            ds = Dataset(t_cov)

            dataset = ClairvoyanceTorchDataset(ds)

            assert len(dataset) == 2
            for item in dataset:
                assert len(item) == 4
                t_cov, s_cov, _, _ = item
                assert np.isnan(s_cov).all()
                assert isinstance(t_cov, torch.Tensor)
                assert tuple(t_cov.shape) == (4, 3)

        def test_temporal_and_static_covariates(self):
            t_cov = TimeSeriesSamples(
                [
                    np.asarray([[1.0, 2.0, 3.0, 4.0], [11.0, 12.0, 13.0, 14.0], [111.0, 112.0, 113.0, 114.0]]).T,
                    np.asarray(
                        [[-1.0, -2.0, -3.0, -4.0], [-11.0, -12.0, -13.0, -14.0], [-111.0, -112.0, -113.0, -114.0]]
                    ).T,
                ]
            )
            s_cov = StaticSamples(np.asarray([np.asarray([99.0, 0.0, 1.0]), np.asarray([121.0, 0.0, 0.0])]))
            ds = Dataset(t_cov, s_cov)

            dataset = ClairvoyanceTorchDataset(ds)

            assert len(dataset) == 2
            for item in dataset:
                assert len(item) == 4
                t_cov, s_cov, _, _ = item
                assert isinstance(t_cov, torch.Tensor)
                assert tuple(t_cov.shape) == (4, 3)
                assert isinstance(s_cov, torch.Tensor)
                assert tuple(s_cov.shape) == (3,)

        def test_temporal_covariates_and_targets(self):
            t_cov = TimeSeriesSamples(
                [
                    np.asarray([[1.0, 2.0, 3.0, 4.0], [11.0, 12.0, 13.0, 14.0], [111.0, 112.0, 113.0, 114.0]]).T,
                    np.asarray(
                        [[-1.0, -2.0, -3.0, -4.0], [-11.0, -12.0, -13.0, -14.0], [-111.0, -112.0, -113.0, -114.0]]
                    ).T,
                ]
            )
            t_targ = TimeSeriesSamples(
                [
                    np.asarray([[1.0, 1.0, 0.0, 0.0]]).T,
                    np.asarray([[0.0, 0.0, 1.0, 1.0]]).T,
                ]
            )
            ds = Dataset(temporal_covariates=t_cov, temporal_targets=t_targ)

            dataset = ClairvoyanceTorchDataset(ds)

            assert len(dataset) == 2
            for item in dataset:
                assert len(item) == 4
                t_cov, s_cov, t_targ, _ = item
                assert np.isnan(s_cov).all()
                assert isinstance(t_cov, torch.Tensor)
                assert tuple(t_cov.shape) == (4, 3)
                assert isinstance(t_targ, torch.Tensor)
                assert tuple(t_targ.shape) == (4, 1)

        def test_temporal_covariates_targets_and_treatments(self):
            t_cov = TimeSeriesSamples(
                [
                    np.asarray([[1.0, 2.0, 3.0, 4.0], [11.0, 12.0, 13.0, 14.0], [111.0, 112.0, 113.0, 114.0]]).T,
                    np.asarray(
                        [[-1.0, -2.0, -3.0, -4.0], [-11.0, -12.0, -13.0, -14.0], [-111.0, -112.0, -113.0, -114.0]]
                    ).T,
                ]
            )
            t_targ = TimeSeriesSamples(
                [
                    np.asarray([[1.0, 1.0, 0.0, 0.0]]).T,
                    np.asarray([[0.0, 0.0, 1.0, 1.0]]).T,
                ]
            )
            t_treat = TimeSeriesSamples(
                [
                    np.asarray([[1.0, 0.0, 1.0, 0.0]]).T,
                    np.asarray([[1.0, 1.0, 1.0, 0.0]]).T,
                ]
            )
            ds = Dataset(temporal_covariates=t_cov, temporal_targets=t_targ, temporal_treatments=t_treat)

            dataset = ClairvoyanceTorchDataset(ds)

            assert len(dataset) == 2
            for item in dataset:
                assert len(item) == 4
                t_cov, s_cov, t_targ, t_treat = item
                assert np.isnan(s_cov).all()
                assert isinstance(t_cov, torch.Tensor)
                assert tuple(t_cov.shape) == (4, 3)
                assert isinstance(t_targ, torch.Tensor)
                assert tuple(t_targ.shape) == (4, 1)
                assert isinstance(t_treat, torch.Tensor)
                assert tuple(t_treat.shape) == (4, 1)

        def test_temporal_covariates_targets_treatments_and_static_covariates(self):
            t_cov = TimeSeriesSamples(
                [
                    np.asarray([[1.0, 2.0, 3.0, 4.0], [11.0, 12.0, 13.0, 14.0], [111.0, 112.0, 113.0, 114.0]]).T,
                    np.asarray(
                        [[-1.0, -2.0, -3.0, -4.0], [-11.0, -12.0, -13.0, -14.0], [-111.0, -112.0, -113.0, -114.0]]
                    ).T,
                ]
            )
            s_cov = StaticSamples(np.asarray([np.asarray([99.0, 0.0, 1.0]), np.asarray([121.0, 0.0, 0.0])]))
            t_targ = TimeSeriesSamples(
                [
                    np.asarray([[1.0, 1.0, 0.0, 0.0]]).T,
                    np.asarray([[0.0, 0.0, 1.0, 1.0]]).T,
                ]
            )
            t_treat = TimeSeriesSamples(
                [
                    np.asarray([[1.0, 0.0, 1.0, 0.0]]).T,
                    np.asarray([[1.0, 1.0, 1.0, 0.0]]).T,
                ]
            )
            ds = Dataset(
                temporal_covariates=t_cov, static_covariates=s_cov, temporal_targets=t_targ, temporal_treatments=t_treat
            )

            dataset = ClairvoyanceTorchDataset(ds)

            assert len(dataset) == 2
            for item in dataset:
                assert len(item) == 4
                t_cov, s_cov, t_targ, t_treat = item
                assert isinstance(t_cov, torch.Tensor)
                assert tuple(t_cov.shape) == (4, 3)
                assert isinstance(s_cov, torch.Tensor)
                assert tuple(s_cov.shape) == (3,)
                assert isinstance(t_targ, torch.Tensor)
                assert tuple(t_targ.shape) == (4, 1)
                assert isinstance(t_treat, torch.Tensor)
                assert tuple(t_treat.shape) == (4, 1)

    class TestToCustomTorchDatasetFunction:
        def test_static_and_time_series_samples_containers(self, a, b):
            a = TimeSeriesSamples(data=a)
            b = StaticSamples(data=b[:, :, 0])

            dataset = CustomTorchDataset(containers=[a, b])
            (get_item_a, get_item_b) = dataset[4]

            assert dataset.index_dimensions == [0, 0]
            assert isinstance(get_item_a, torch.Tensor)
            assert isinstance(get_item_b, torch.Tensor)
            assert (get_item_a == torch.tensor([[5.0, 10.0, 15.0], [50.0, 100.0, 150.0]])).all()
            assert (get_item_b == torch.tensor([[-5.0, -50.0]])).all()

        def test_static_and_time_series_samples_containers_with_nones(self, a, b):
            a = TimeSeriesSamples(data=a)
            b = StaticSamples(data=b[:, :, 0])

            dataset = CustomTorchDataset(containers=[a, None, b, None])
            (get_item_a, get_item_n1, get_item_b, get_item_n2) = dataset[4]

            assert dataset.index_dimensions == [0, 0, 0, 0]
            assert isinstance(get_item_a, torch.Tensor)
            assert isinstance(get_item_n1, torch.Tensor)
            assert isinstance(get_item_b, torch.Tensor)
            assert isinstance(get_item_n2, torch.Tensor)
            assert (get_item_a == torch.tensor([[5.0, 10.0, 15.0], [50.0, 100.0, 150.0]])).all()
            assert (get_item_b == torch.tensor([[-5.0, -50.0]])).all()
            assert get_item_n1.isnan().all().item() is True
            assert get_item_n2.isnan().all().item() is True


class TestToCustomTorchDatasetFunction:
    class TestFunctionalityUsingArrays:
        def test_single_array_iteration(self, a):
            dataset = CustomTorchDataset(containers=[a])
            (get_item,) = dataset[4]  # Note the comma.

            assert dataset.index_dimensions == [0]
            assert isinstance(get_item, torch.Tensor)
            assert (get_item == torch.tensor([[5.0, 10.0, 15.0], [50.0, 100.0, 150.0]])).all()

        def test_single_array_iteration_specify_index_dim(self):
            a = np.asarray([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
            a = np.asarray([a * (idx + 1) for idx in range(4)])

            dataset = CustomTorchDataset(containers=[a], index_dimensions=[2])
            (get_item,) = dataset[1]

            assert dataset.index_dimensions == [2]
            assert isinstance(get_item, torch.Tensor)
            assert (get_item == torch.tensor([[2.0, 20.0], [4.0, 40.0], [6.0, 60.0], [8.0, 80.0]])).all()

        def test_single_array_iteration_specify_index_dim_with_none(self):
            a = np.asarray([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
            a = np.asarray([a * (idx + 1) for idx in range(4)])

            dataset = CustomTorchDataset(containers=[a, None], index_dimensions=[2, 0])
            (get_item, get_item_none) = dataset[1]

            assert dataset.index_dimensions == [2, 0]
            assert isinstance(get_item, torch.Tensor)
            assert (get_item == torch.tensor([[2.0, 20.0], [4.0, 40.0], [6.0, 60.0], [8.0, 80.0]])).all()
            assert get_item_none.isnan().all().item() is True

        def test_multiple_arrays_iteration(self, a, b):
            dataset = CustomTorchDataset(containers=[a, b])
            (get_item_a, get_item_b) = dataset[4]

            assert dataset.index_dimensions == [0, 0]
            assert isinstance(get_item_a, torch.Tensor)
            assert isinstance(get_item_b, torch.Tensor)
            assert (get_item_a == torch.tensor([[5.0, 10.0, 15.0], [50.0, 100.0, 150.0]])).all()
            assert (get_item_b == torch.tensor([[-5.0, -10.0, -15.0], [-50.0, -100.0, -150.0]])).all()

        def test_len(self, a, b):
            dataset = CustomTorchDataset(containers=[a, b])
            assert len(dataset) == 10

    def test_torch_tensor_containers(self, a, b):
        a = torch.tensor(a)
        b = torch.tensor(b)

        dataset = CustomTorchDataset(containers=[a, b])
        (get_item_a, get_item_b) = dataset[4]

        assert dataset.index_dimensions == [0, 0]
        assert isinstance(get_item_a, torch.Tensor)
        assert isinstance(get_item_b, torch.Tensor)
        assert (get_item_a == torch.tensor([[5.0, 10.0, 15.0], [50.0, 100.0, 150.0]])).all()
        assert (get_item_b == torch.tensor([[-5.0, -10.0, -15.0], [-50.0, -100.0, -150.0]])).all()

    class TestValidation:
        def test_fail_no_containers(self):
            with pytest.raises(ValueError) as excinfo:
                CustomTorchDataset(containers=[])
            assert "at least one" in str(excinfo.value)

        def test_fail_index_dimensions_arg_wrong_len(self):
            with pytest.raises(ValueError) as excinfo:
                CustomTorchDataset(containers=[Mock(), Mock()], index_dimensions=[1, 1, 0])
            assert "must be the same as the length" in str(excinfo.value)

        def test_fail_incorrect_index_dimension_specified_for_static_samples(self):
            with pytest.raises(ValueError) as excinfo:
                CustomTorchDataset(containers=[Mock(spec=StaticSamples)], index_dimensions=[1])
            assert "by 0th dimension" in str(excinfo.value)

        def test_fail_incorrect_index_dimension_specified_for_time_series_samples(self):
            with pytest.raises(ValueError) as excinfo:
                CustomTorchDataset(containers=[Mock(spec=TimeSeriesSamples)], index_dimensions=[2])
            assert "by 0th dimension" in str(excinfo.value)

        def test_fail_all_containers_none(self):
            with pytest.raises(ValueError) as excinfo:
                CustomTorchDataset(containers=[None, None], index_dimensions=[2])
            assert "None" in str(excinfo.value)

        @pytest.mark.parametrize(
            "container",
            [
                Mock(spec=np.ndarray, ndim=3),
                Mock(spec=TimeSeriesSamples),
                Mock(spec=StaticSamples),
            ],
        )
        def test_fail_index_dimensions_too_high(self, container):
            with pytest.raises(ValueError) as excinfo:
                CustomTorchDataset(containers=[container], index_dimensions=[4])
            assert "exceeded the number of dimensions" in str(excinfo.value)

        def test_fail_raise_wrong_container_type(self):
            class MyRandomClass:
                pass

            with pytest.raises(ValueError) as excinfo:
                CustomTorchDataset(containers=[Mock(spec=MyRandomClass)])
            assert "`container` was not one of types" in str(excinfo.value)

        def test_fail_index_dimensions_len_mismatch(self, a):
            b = np.asarray([[-1.0, -2.0, -3.0], [-10.0, -20.0, -30.0]])
            b = np.asarray([b * (idx + 1) for idx in range(5)])

            with pytest.raises(ValueError) as excinfo:
                CustomTorchDataset(containers=[a, b], index_dimensions=[0, 0])
            assert "same length along the index dimension" in str(excinfo.value)
