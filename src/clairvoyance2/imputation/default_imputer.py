import pandas as pd

from ..data import FeatureType, TDataset, TimeSeries, TimeSeriesSamples
from ..interface import BaseModel, TransformerMixin
from ..utils.dev import raise_not_implemented

# pylint: disable=attribute-defined-outside-init
# ^ Expected, as .fit() sets parameters.


# TODO: Integration and unit tests.
class DefaultTimeSeriesSamplesImputer(BaseModel, TransformerMixin):
    def __init__(self) -> None:
        super().__init__(dict())

    def validate_data(self, data: TDataset) -> None:
        # Will be eventually handled by Requirements.
        assert isinstance(data, TimeSeriesSamples)
        if any([f.feature_type != FeatureType.NUMERIC for f in data.features.values()]):
            raise_not_implemented(feature="DefaultTimeSeriesSamplesImputer imputation for non-numeric features")

    def train(self, data: TDataset) -> "DefaultTimeSeriesSamplesImputer":
        data = self.parse_data_argument(data)
        self.validate_data(data)

        assert isinstance(data, TimeSeriesSamples)  # For mypy
        df_global = pd.concat([x.df for x in data], axis=0, ignore_index=True)
        means = df_global.mean()
        if means.isnull().values.any():
            problem_cols = list(means[means.isnull()].index)
            raise ValueError(
                "Found that the following columns in the time series data "
                f"are completely absent across all samples: {problem_cols}, cannot impute"
            )
        self.global_fill_values = means.to_dict()

        return self

    def _impute_single_timeseries(self, timeseries: TimeSeries) -> TimeSeries:
        timeseries.df = timeseries.df.ffill(axis=0).bfill(axis=0).fillna(value=self.global_fill_values)
        return timeseries

    def transform(self, data: TDataset) -> TDataset:
        data = self.parse_data_argument(data)
        self.validate_data(data)

        assert isinstance(data, TimeSeriesSamples)  # For mypy
        list_ts = []
        for ts in data:
            list_ts.append(self._impute_single_timeseries(ts))

        new_data = TimeSeriesSamples(
            list_ts,
            categorical_features=data._categorical_def,  # pylint: disable=protected-access
            missing_indicator=data.missing_indicator,
        )
        assert not new_data.has_missing
        return new_data
