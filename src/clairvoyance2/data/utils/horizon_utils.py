from typing import Callable, List, Sequence, Union

from ...interface.horizon import TimeIndexHorizon
from .. import Dataset, TimeSeries, TimeSeriesSamples
from ..constants import T_TSIndexClass

_TPerTimeSeriesFunction = Callable[[TimeSeries, T_TSIndexClass, bool, List[TimeSeries]], List[TimeSeries]]


# TODO: Test.


def _take_all_before_horizon_start(
    time_series: TimeSeries, time_index: T_TSIndexClass, inplace: bool, list_ts: List[TimeSeries]
) -> List[TimeSeries]:
    take_up_to_index = time_series.time_index[time_series.time_index < time_index[0]][-1]
    if inplace is True:
        time_series.apply_time_indexing(slice(None, take_up_to_index), inplace=True)
    else:
        ts_new = time_series.apply_time_indexing(slice(None, take_up_to_index), inplace=False)
        assert isinstance(ts_new, TimeSeries)
        list_ts.append(ts_new)
    return list_ts


def _take_one_before_horizon_start(
    time_series: TimeSeries, time_index: T_TSIndexClass, inplace: bool, list_ts: List[TimeSeries]
) -> List[TimeSeries]:
    take_at_index = time_series.time_index[time_series.time_index < time_index[0]][-1]
    if inplace is True:
        time_series.apply_time_indexing([take_at_index], inplace=True)
    else:
        ts_new = time_series.apply_time_indexing([take_at_index], inplace=False)
        assert isinstance(ts_new, TimeSeries)
        list_ts.append(ts_new)
    return list_ts


def _take_all_from_horizon_start(
    time_series: TimeSeries, time_index: T_TSIndexClass, inplace: bool, list_ts: List[TimeSeries]
) -> List[TimeSeries]:
    take_from_index_on = time_series.time_index[time_series.time_index >= time_index[0]][0]
    if inplace is True:
        time_series.apply_time_indexing(slice(take_from_index_on, None), inplace=True)
    else:
        ts_new = time_series.apply_time_indexing(slice(take_from_index_on, None), inplace=False)
        assert isinstance(ts_new, TimeSeries)
        list_ts.append(ts_new)
    return list_ts


def _take_all_from_one_before_horizon_start(
    time_series: TimeSeries, time_index: T_TSIndexClass, inplace: bool, list_ts: List[TimeSeries]
) -> List[TimeSeries]:
    take_from_index_on = time_series.time_index[time_series.time_index < time_index[0]][-1]
    if inplace is True:
        time_series.apply_time_indexing(slice(take_from_index_on, None), inplace=True)
    else:
        ts_new = time_series.apply_time_indexing(slice(take_from_index_on, None), inplace=False)
        assert isinstance(ts_new, TimeSeries)
        list_ts.append(ts_new)
    return list_ts


class time_series_samples:
    @staticmethod
    def _horizon_interaction_time_series_samples(
        time_series_samples_: TimeSeriesSamples,
        horizon: TimeIndexHorizon,
        inplace: bool,
        per_time_series_method: _TPerTimeSeriesFunction,
    ):
        if len(horizon.time_index_sequence) != time_series_samples_.n_samples:
            raise ValueError(
                "Time index sequence specified in the time index horizon must be the same length as the "
                "number of samples in the time series samples object, but was "
                f"{len(horizon.time_index_sequence)} and {time_series_samples_.n_samples} respectively"
            )
        list_ts: List[TimeSeries] = []
        for time_series, time_index in zip(time_series_samples_, horizon.time_index_sequence):
            list_ts = per_time_series_method(time_series, time_index, inplace, list_ts)
        if inplace is False:
            return TimeSeriesSamples.new_like(time_series_samples_, data=list_ts)

    @staticmethod
    def take_all_before_horizon_start(
        time_series_samples_: TimeSeriesSamples, horizon: TimeIndexHorizon, inplace: bool = False
    ):
        return time_series_samples._horizon_interaction_time_series_samples(
            time_series_samples_, horizon, inplace, per_time_series_method=_take_all_before_horizon_start
        )

    @staticmethod
    def take_one_before_horizon_start(
        time_series_samples_: TimeSeriesSamples, horizon: TimeIndexHorizon, inplace: bool = False
    ):
        return time_series_samples._horizon_interaction_time_series_samples(
            time_series_samples_, horizon, inplace, per_time_series_method=_take_one_before_horizon_start
        )

    @staticmethod
    def take_all_from_horizon_start(
        time_series_samples_: TimeSeriesSamples, horizon: TimeIndexHorizon, inplace: bool = False
    ):
        return time_series_samples._horizon_interaction_time_series_samples(
            time_series_samples_, horizon, inplace, per_time_series_method=_take_all_from_horizon_start
        )

    @staticmethod
    def take_all_from_one_before_horizon_start(
        time_series_samples_: TimeSeriesSamples, horizon: TimeIndexHorizon, inplace: bool = False
    ):
        return time_series_samples._horizon_interaction_time_series_samples(
            time_series_samples_,
            horizon,
            inplace,
            per_time_series_method=_take_all_from_one_before_horizon_start,
        )


class dataset:
    @staticmethod
    def take_temporal_data_before_horizon_start(
        data: Dataset, horizon: TimeIndexHorizon, ignore_containers: Sequence[str] = tuple(), inplace: bool = False
    ) -> Union[Dataset, None]:
        if inplace is False:
            data = data.copy()

        if len(horizon.time_index_sequence) != data.n_samples:
            raise ValueError(
                "Time index sequence specified in the time index horizon must be the same length as the "
                "number of samples in the dataset but was "
                f"{len(horizon.time_index_sequence)} and {data.n_samples} respectively"
            )

        for container_name, container in data.temporal_data_containers.items():
            skip = False
            if len(ignore_containers) > 0 and container_name in ignore_containers:
                skip = True
            if not skip:
                time_series_samples.take_all_before_horizon_start(container, horizon, inplace=True)

        if inplace is False:
            return data
        else:
            return None
