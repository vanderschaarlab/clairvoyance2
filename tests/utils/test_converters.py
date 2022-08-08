import numpy as np
import torch

from clairvoyance2.data import Dataset, StaticSamples, TimeSeriesSamples
from clairvoyance2.utils.converters import to_torch_dataset


class TestIntegration:
    class TestToTorchDatasetFunction:
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

            dataset = to_torch_dataset(ds)

            assert len(dataset) == 2
            for item in dataset:
                assert len(item) == 3
                t_cov, _, s_cov = item
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

            dataset = to_torch_dataset(ds)

            assert len(dataset) == 2
            for item in dataset:
                assert len(item) == 3
                t_cov, _, s_cov = item
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
            t_lab = TimeSeriesSamples(
                [
                    np.asarray([[1.0, 1.0, 0.0, 0.0]]).T,
                    np.asarray([[0.0, 0.0, 1.0, 1.0]]).T,
                ]
            )
            ds = Dataset(temporal_covariates=t_cov, temporal_targets=t_lab)

            dataset = to_torch_dataset(ds)

            assert len(dataset) == 2
            for item in dataset:
                assert len(item) == 3
                t_cov, t_lab, s_cov = item
                assert np.isnan(s_cov).all()
                assert isinstance(t_cov, torch.Tensor)
                assert tuple(t_cov.shape) == (4, 3)
                assert isinstance(t_lab, torch.Tensor)
                assert tuple(t_lab.shape) == (4, 1)
