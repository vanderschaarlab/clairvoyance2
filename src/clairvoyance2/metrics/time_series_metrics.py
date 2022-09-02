import numpy as np
import pandas as pd

from ..data import Dataset, TimeSeries
from ..data.constants import T_NumericDtype_AsTuple
from ..utils.common import python_type_from_np_pd_dtype
from .metric import BaseMetric, MetricFor, TMetricValue


# Abbreviation: TS = Time Series
class BaseMetricTS(BaseMetric):
    def __init__(self) -> None:
        super().__init__(metric_for=MetricFor.TEMPORAL_TARGETS)


def squared_error(arr_true: np.ndarray, arr_pred: np.ndarray) -> np.ndarray:
    assert arr_true.shape == arr_pred.shape
    return (arr_true - arr_pred) ** 2


# TODO: Multiple types of output - per sample, per time-step sample, overall value.
class MSEMetricTS(BaseMetricTS):
    def __call__(self, data_true: Dataset, data_pred: Dataset) -> TMetricValue:

        # TODO: All this will be moved to Metric requirements.
        assert data_true.temporal_targets is not None
        assert data_pred.temporal_targets is not None

        acceptable_types = T_NumericDtype_AsTuple
        ts = data_true.temporal_targets[0]
        assert isinstance(ts, TimeSeries)
        assert python_type_from_np_pd_dtype(ts.time_index.dtype) in acceptable_types
        assert python_type_from_np_pd_dtype(ts.time_index.dtype) in acceptable_types
        # TODO: ^ Make a convenience method to get the above `dtype` easily.

        assert list(data_true.temporal_targets.feature_names) == list(data_pred.temporal_targets.feature_names)

        assert data_true.temporal_targets.sample_indices == data_pred.temporal_targets.sample_indices

        list_metric_dfs = []
        for ts_true, ts_pred in zip(data_true.temporal_targets, data_pred.temporal_targets):
            df_true: pd.DataFrame = ts_true.df
            df_pred: pd.DataFrame = ts_pred.df
            df_true_aligned, df_pred_aligned = df_true.align(df_pred, join="inner", axis=0)
            assert len(df_true_aligned) > 0

            metric = squared_error(df_true_aligned.values, df_pred_aligned.values)
            list_metric_dfs.append(
                pd.DataFrame(data=metric, index=df_true_aligned.index, columns=df_true_aligned.columns)
            )

        multi_index_df = pd.concat(list_metric_dfs, axis=0, keys=data_true.temporal_targets.sample_indices)
        feature_mean_metric = multi_index_df.mean()
        overall_mean_metric = feature_mean_metric.mean()

        return overall_mean_metric


class RMSEMetricTS(MSEMetricTS):
    def __call__(self, data_true: Dataset, data_pred: Dataset) -> TMetricValue:
        return super().__call__(data_true, data_pred) ** (1 / 2)


mse_ts = MSEMetricTS()
rmse_ts = RMSEMetricTS()
