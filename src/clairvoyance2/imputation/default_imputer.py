import pandas as pd

from ..data import Dataset, TimeSeries, TimeSeriesSamples
from ..interface import TransformerModel
from ..interface.requirements import DatasetRequirements, Requirements

# pylint: disable=attribute-defined-outside-init
# ^ Expected, as .fit() sets parameters.


# TODO: Unit tests.
# NOTE: This imputer does nothing to temporal *targets*, only affects temporal *covariates*.
class DefaultImputerTC(TransformerModel):
    requirements: Requirements = Requirements(
        dataset_requirements=DatasetRequirements(requires_all_numeric_features=True),
        prediction_requirements=None,  # Transformers do not have prediction requirements.
    )

    def _fit(self, data: Dataset) -> "DefaultImputerTC":
        temporal_covariates, _, _ = data

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

    def _transform(self, data: Dataset) -> Dataset:
        data = data.copy()
        temporal_covariates, _, _ = data

        assert isinstance(temporal_covariates, TimeSeriesSamples)  # For mypy
        list_ts = []
        for ts in temporal_covariates:
            list_ts.append(self._impute_single_timeseries(ts))

        new_temporal_covariates = TimeSeriesSamples(
            list_ts,
            categorical_features=temporal_covariates._categorical_def,  # pylint: disable=protected-access
            missing_indicator=temporal_covariates.missing_indicator,
        )
        assert not new_temporal_covariates.has_missing

        data.temporal_covariates = new_temporal_covariates
        return data
