import pandas as pd

from ..data import Dataset, TimeSeries, TimeSeriesSamples
from ..interface import TransformerModel
from ..interface.requirements import DatasetRequirements, DataValueOpts, Requirements

# pylint: disable=attribute-defined-outside-init
# ^ Expected, as .fit() sets parameters.


# TODO: Unit tests.
# NOTE: This imputer does nothing to temporal *targets*, only affects temporal *covariates*.
class TemporalDataDefaultImputer(TransformerModel):
    requirements: Requirements = Requirements(
        dataset_requirements=DatasetRequirements(temporal_covariates_value_type=DataValueOpts.NUMERIC),
        prediction_requirements=None,  # Transformers do not have prediction requirements.
    )

    def _fit(self, data: Dataset, **kwargs) -> "TemporalDataDefaultImputer":
        temporal_covariates = data.temporal_covariates

        assert isinstance(temporal_covariates, TimeSeriesSamples)  # For mypy
        df_global = pd.concat([x.df for x in temporal_covariates], axis=0, ignore_index=True)
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

    def _transform(self, data: Dataset, **kwargs) -> Dataset:
        data = data.copy()
        temporal_covariates = data.temporal_covariates

        assert isinstance(temporal_covariates, TimeSeriesSamples)  # For mypy
        list_ts = []
        for ts in temporal_covariates:
            list_ts.append(self._impute_single_timeseries(ts))

        new_temporal_covariates = TimeSeriesSamples.new_like(like=temporal_covariates, data=list_ts)
        assert not new_temporal_covariates.has_missing

        data.temporal_covariates = new_temporal_covariates
        return data
