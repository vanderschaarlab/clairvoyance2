from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from clairvoyance2.data import Dataset, StaticSamples, TimeSeriesSamples
from clairvoyance2.preprocessing import (
    MinMaxScalerSC,
    MinMaxScalerTC,
    SklearnTransformerSC,
    SklearnTransformerTC,
    StandardScalerSC,
    StandardScalerTC,
)

# pylint: disable=redefined-outer-name
# ^ Otherwise pylint trips up on pytest fixtures.


@pytest.fixture
def three_numeric_dfs():
    df_0 = pd.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})
    df_1 = pd.DataFrame({"a": [7, 8, 9], "b": [7.0, 8.0, 9.0]})
    df_2 = pd.DataFrame({"a": [-1, -2, -3], "b": [11.0, 12.0, 13.0]})
    return (df_0, df_1, df_2)


class TestIntegration:
    class TestSklearnTransformerSC:
        def test_init_and_all_methods_success(self):
            # Arrange.
            my_transformer = SklearnTransformerSC(
                sklearn_transformer=StandardScaler, params=dict(copy=True, with_mean=True, with_std=False)
            )
            static_covariates = StaticSamples(np.asarray([[1.0, 12.0, 18.0], [0.0, -5.0, 22.0], [11.0, 2.0, 97.0]]))
            temporal_covariates = Mock(spec=TimeSeriesSamples, n_samples=3)
            data = Dataset(temporal_covariates=temporal_covariates, static_covariates=static_covariates)

            # Act.
            my_transformer: SklearnTransformerSC = my_transformer.fit(data)
            data_transformed = my_transformer.transform(data)
            data_transformed = my_transformer.fit_transform(data)
            data_inverse_transformed = my_transformer.inverse_transform(  # noqa  # pylint: disable=unused-variable
                data_transformed
            )

            # Assert no errors.

    class TestStandardScalerSC:
        def test_transform_inverse_transform(self):
            scaler = StandardScalerSC(params=dict())
            static_covariates = StaticSamples(np.asarray([[1.0, 2.0, 3.0], [0.0, -5.0, 5.0], [-10.0, 20.0, -30.0]]))
            temporal_covariates = Mock(spec=TimeSeriesSamples, n_samples=3)
            data = Dataset(temporal_covariates=temporal_covariates, static_covariates=static_covariates)

            data_transformed = scaler.fit_transform(data)
            data_inverse_transformed = scaler.inverse_transform(data_transformed)

            assert np.allclose(
                data_transformed.static_covariates.df.values,
                np.asarray(
                    [
                        [0.80538727, -0.34819892, 0.64388006],
                        [0.60404045, -1.0129423, 0.76850201],
                        [-1.40942772, 1.36114122, -1.41238207],
                    ]
                ),
            )
            assert np.allclose(
                data_inverse_transformed.static_covariates.df.values,
                np.asarray([[1.0, 2.0, 3.0], [0.0, -5.0, 5.0], [-10.0, 20.0, -30.0]]),
            )
            assert (
                data.static_covariates.df.values
                == np.asarray([[1.0, 2.0, 3.0], [0.0, -5.0, 5.0], [-10.0, 20.0, -30.0]])
            ).all()

    class TestMinMaxScalerSC:
        def test_transform_inverse_transform(self):
            scaler = MinMaxScalerSC(params=dict())
            static_covariates = StaticSamples(np.asarray([[1.0, 2.0, 3.0], [0.0, -5.0, 5.0], [-10.0, 20.0, -30.0]]))
            temporal_covariates = Mock(spec=TimeSeriesSamples, n_samples=3)
            data = Dataset(temporal_covariates=temporal_covariates, static_covariates=static_covariates)

            data_transformed = scaler.fit_transform(data)
            data_inverse_transformed = scaler.inverse_transform(data_transformed)

            assert np.allclose(
                data_transformed.static_covariates.df.values,
                np.asarray([[1.0, 0.28, 0.94285714], [0.90909091, 0.0, 1.0], [0.0, 1.0, 0.0]]),
            )
            assert np.allclose(
                data_inverse_transformed.static_covariates.df.values,
                np.asarray([[1.0, 2.0, 3.0], [0.0, -5.0, 5.0], [-10.0, 20.0, -30.0]]),
            )
            assert (
                data.static_covariates.df.values
                == np.asarray([[1.0, 2.0, 3.0], [0.0, -5.0, 5.0], [-10.0, 20.0, -30.0]])
            ).all()

    class TestSklearnTransformerTC:
        def test_init_and_all_methods_success(self, three_numeric_dfs):
            # Arrange.
            my_transformer = SklearnTransformerTC(
                sklearn_transformer=StandardScaler, params=dict(copy=True, with_mean=True, with_std=False)
            )
            static_covariates = Mock(spec=StaticSamples, n_samples=3)
            temporal_covariates = TimeSeriesSamples(three_numeric_dfs)
            data = Dataset(temporal_covariates=temporal_covariates, static_covariates=static_covariates)

            # Act.
            my_transformer: SklearnTransformerTC = my_transformer.fit(data)
            data_transformed = my_transformer.transform(data)
            data_transformed = my_transformer.fit_transform(data)
            data_inverse_transformed = my_transformer.inverse_transform(  # noqa  # pylint: disable=unused-variable
                data_transformed
            )

            # Assert no errors.

    class TestStandardScalerTC:
        def test_transform_inverse_transform(self, three_numeric_dfs):
            scaler = StandardScalerTC(params=dict())
            static_covariates = Mock(spec=StaticSamples, n_samples=3)
            temporal_covariates = TimeSeriesSamples(three_numeric_dfs)
            data = Dataset(temporal_covariates=temporal_covariates, static_covariates=static_covariates)

            data_transformed = scaler.fit_transform(data)
            data_inverse_transformed = scaler.inverse_transform(data_transformed)

            # Transformed:
            assert np.allclose(
                data_transformed.temporal_covariates[0].df.values,
                np.asarray(
                    [
                        [-0.39777864, -1.51155884],
                        [-0.15911146, -1.27289165],
                        [0.07955573, -1.03422447],
                    ]
                ),
            )
            assert np.allclose(
                data_transformed.temporal_covariates[1].df.values,
                np.asarray(
                    [
                        [1.03422447, -0.07955573],
                        [1.27289165, 0.15911146],
                        [1.51155884, 0.39777864],
                    ]
                ),
            )
            assert np.allclose(
                data_transformed.temporal_covariates[2].df.values,
                np.asarray([[-0.87511301, 0.87511301], [-1.1137802, 1.1137802], [-1.35244738, 1.35244738]]),
            )
            # Inverse transformed:
            assert np.allclose(
                data_inverse_transformed.temporal_covariates[0].df.values,
                np.asarray([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]).T,
            )
            assert np.allclose(
                data_inverse_transformed.temporal_covariates[1].df.values,
                np.asarray([[7.0, 8.0, 9.0], [7.0, 8.0, 9.0]]).T,
            )
            assert np.allclose(
                data_inverse_transformed.temporal_covariates[2].df.values,
                np.asarray([[-1.0, -2.0, -3.0], [11.0, 12.0, 13.0]]).T,
            )
            # Original:
            assert (data.temporal_covariates[0].df == pd.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})).all().all()
            assert (data.temporal_covariates[1].df == pd.DataFrame({"a": [7, 8, 9], "b": [7.0, 8.0, 9.0]})).all().all()
            assert (
                (data.temporal_covariates[2].df == pd.DataFrame({"a": [-1, -2, -3], "b": [11.0, 12.0, 13.0]}))
                .all()
                .all()
            )

    class TestMinMaxScalerTC:
        def test_transform_inverse_transform(self, three_numeric_dfs):
            scaler = MinMaxScalerTC(params=dict())
            static_covariates = Mock(spec=StaticSamples, n_samples=3)
            temporal_covariates = TimeSeriesSamples(three_numeric_dfs)
            data = Dataset(temporal_covariates=temporal_covariates, static_covariates=static_covariates)

            data_transformed = scaler.fit_transform(data)
            data_inverse_transformed = scaler.inverse_transform(data_transformed)

            # Transformed:
            assert np.allclose(
                data_transformed.temporal_covariates[0].df.values,
                np.asarray(
                    [
                        [0.33333333, 0.0],
                        [0.41666667, 0.08333333],
                        [0.5, 0.16666667],
                    ]
                ),
            )
            assert np.allclose(
                data_transformed.temporal_covariates[1].df.values,
                np.asarray(
                    [
                        [0.83333333, 0.5],
                        [0.91666667, 0.58333333],
                        [1.0, 0.66666667],
                    ]
                ),
            )
            assert np.allclose(
                data_transformed.temporal_covariates[2].df.values,
                np.asarray([[0.16666667, 0.83333333], [0.08333333, 0.91666667], [0.0, 1.0]]),
            )
            # Inverse transformed:
            assert np.allclose(
                data_inverse_transformed.temporal_covariates[0].df.values,
                np.asarray([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]).T,
            )
            assert np.allclose(
                data_inverse_transformed.temporal_covariates[1].df.values,
                np.asarray([[7.0, 8.0, 9.0], [7.0, 8.0, 9.0]]).T,
            )
            assert np.allclose(
                data_inverse_transformed.temporal_covariates[2].df.values,
                np.asarray([[-1.0, -2.0, -3.0], [11.0, 12.0, 13.0]]).T,
            )
            # Original:
            assert (data.temporal_covariates[0].df == pd.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})).all().all()
            assert (data.temporal_covariates[1].df == pd.DataFrame({"a": [7, 8, 9], "b": [7.0, 8.0, 9.0]})).all().all()
            assert (
                (data.temporal_covariates[2].df == pd.DataFrame({"a": [-1, -2, -3], "b": [11.0, 12.0, 13.0]}))
                .all()
                .all()
            )
