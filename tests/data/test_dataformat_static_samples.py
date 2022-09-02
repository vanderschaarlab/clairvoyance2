import numpy as np
import pandas as pd
import pytest

from clairvoyance2.data.dataformat import StaticSamples
from clairvoyance2.data.feature import FeatureType

# pylint: disable=redefined-outer-name
# ^ Otherwise pylint trips up on pytest fixtures.


@pytest.fixture
def df_numeric():
    return pd.DataFrame({"col_1": [1.0, 2.0, 3.0], "col_2": [1, 2, 2]})


# NOTE: For now, only integration tests that make sure the whole dataformat ecosystem works.
class TestIntegration:
    def test_init(self, df_numeric):
        StaticSamples(df_numeric)

    @pytest.mark.parametrize("sample_indices", [[3, 4, 8], [0, 1, 2]])
    @pytest.mark.parametrize(
        "data",
        [
            pd.DataFrame({"a": [1, 2, 3], "b": [3.0, 9.0, 1.0]}),
            np.zeros(shape=(3, 5)),
        ],
    )
    def test_init_success_with_sample_indices(self, data, sample_indices):
        ss = StaticSamples(data=data, sample_indices=sample_indices)

        assert len(ss) == 3
        assert ss.empty is False
        assert ss.sample_indices == sample_indices
        assert list(ss.df.index) == sample_indices
        assert list(ss.sample_index) == sample_indices

    def test_init_success_empty(self):
        ss = StaticSamples(data=pd.DataFrame({"a": []}))
        assert ss.empty is True
        assert len(ss) == 0

    def test_init_fails_sample_indices_wrong_length(self):
        data = (pd.DataFrame({"a": [1, 2, 3], "b": [3.0, 9.0, 1.0]}),)
        sample_indices = [1, 7, 9, 11]

        with pytest.raises(ValueError) as excinfo:
            StaticSamples(data=data, sample_indices=sample_indices)
        assert "did not match" in str(excinfo.value).lower() and "StaticSamples" in str(excinfo.value)

    @pytest.mark.parametrize(
        "df, expected_result",
        [
            (pd.DataFrame({"col_1": [1.0, np.nan, 3.0], "col_2": [1, 2, np.nan]}), True),
            (pd.DataFrame({"col_1": [1.0, 2.0, 3.0], "col_2": [1, 2, 3]}), False),
        ],
    )
    def test_has_missing(self, df, expected_result):
        ss = StaticSamples(df)
        has_missing = ss.has_missing
        assert has_missing is expected_result

    # --- Indexing-related ---

    def test_len(self, df_numeric):
        ss = StaticSamples(df_numeric)
        length = len(ss)
        assert length == 3

    class TestGetItem:
        def test_single_indexer_item(self):
            data = pd.DataFrame({"col_1": [1.0, 2.0, 3.0], "col_2": [1, 2, 3]})
            ss = StaticSamples(data)

            ss_0 = ss[0]
            ss_2 = ss[2]

            assert isinstance(ss_0, StaticSamples)
            assert isinstance(ss_2, StaticSamples)
            assert (ss_0.df.values == [1.0, 1]).all()
            assert (ss_2.df.values == [3.0, 3]).all()

        def test_single_indexer_slice(self):
            data = pd.DataFrame({"col_1": [1.0, 2.0, 3.0], "col_2": [1, 2, 3]})
            ss = StaticSamples(data)

            sliced = ss[2:]

            assert isinstance(sliced, StaticSamples)
            assert len(sliced.df) == 1

        def test_single_indexer_iterable(self):
            data = pd.DataFrame({"col_1": [1.0, 2.0, 3.0], "col_2": [1, 2, 3]})
            ss = StaticSamples(data)

            ts_new = ss[(0, 2)]

            assert isinstance(ts_new, StaticSamples)
            assert len(ts_new.df) == 2
            assert (ts_new.df == pd.DataFrame({"col_1": [1.0, 3.0], "col_2": [1, 3]}, index=[0, 2])).all().all()

        def test_two_indexers_slice_slice(self):
            data = pd.DataFrame({"col_1": [1.0, 2.0, 3.0], "col_2": [1, 2, 3]})
            ss = StaticSamples(data)

            sliced = ss[1:, "col_2":]

            assert len(sliced.df) == 2
            assert list(sliced.df.index) == [1, 2]
            assert list(sliced.sample_index) == [1, 2]
            assert list(sliced.df.columns) == ["col_2"]

        def test_two_indexers_slice_iterable(self):
            data = pd.DataFrame({"col_1": [1.0, 2.0, 3.0], "col_2": [1, 2, 3], "col_3": [11, 12, 13]})
            ss = StaticSamples(data)

            sliced = ss[1:, ("col_1", "col_2")]

            assert len(sliced.df) == 2
            assert list(sliced.df.index) == [1, 2]
            assert list(sliced.sample_index) == [1, 2]
            assert list(sliced.df.columns) == ["col_1", "col_2"]

    def test_iter(self):
        data = pd.DataFrame({"col_1": [1.0, 2.0, 3.0], "col_2": [1, 2, 3]})
        ss = StaticSamples(data)

        items = []
        for s in ss:
            items.append(s)

        assert all(isinstance(i, StaticSamples) for i in items)
        assert (items[0].df.values == [1.0, 1]).all()
        assert (items[2].df.values == [3.0, 3]).all()

    def test_reversed(self):
        data = pd.DataFrame({"col_1": [1.0, 2.0, 3.0], "col_2": [1, 2, 3]})
        ss = StaticSamples(data)

        items = []
        for s in reversed(ss):
            items.append(s)

        assert all(isinstance(i, StaticSamples) for i in items)
        assert (items[0].df.values == [3.0, 3]).all()
        assert (items[2].df.values == [1.0, 1]).all()

    def test_contains(self):
        data = pd.DataFrame({"col_1": [1.0, 2.0, 3.0], "col_2": [1, 2, 3]})
        ss = StaticSamples(data)

        is_1_in_ts = 1 in ss
        is_9_in_ts = 9 in ss

        assert is_1_in_ts is True
        assert is_9_in_ts is False

    def test_py_sequence_some_methods_not_implemented(self, df_numeric):
        ss = StaticSamples(df_numeric)

        with pytest.raises(NotImplementedError):
            ss.count(None)

        with pytest.raises(NotImplementedError):
            ss.index(None)

    # --- Indexing-related (End) ---

    def test_n_samples(self, df_numeric):
        ss = StaticSamples(df_numeric)
        n_samples = ss.n_samples
        assert n_samples == 3

    def test_features(self):
        data = pd.DataFrame({"col_1": [1.0, 2.0, 3.0], "col_2": ["a", "b", "c"]})
        categorical_features = ["col_2"]
        ss = StaticSamples(data=data, categorical_features=categorical_features)

        features = ss.features

        assert isinstance(features, dict)
        assert len(features) == 2
        assert features["col_1"].feature_type == FeatureType.NUMERIC
        assert features["col_2"].feature_type == FeatureType.CATEGORICAL

    def test_to_numpy(self):
        data = pd.DataFrame({"col_1": [1.0, 2.0, 3.0], "col_2": [11.0, 22.0, 33.0]})
        ss = StaticSamples(data)

        array = ss.to_numpy()

        assert (ss.df.values == array).all()

    def test_copy(self):
        data = pd.DataFrame({"col_1": [1.0, 2.0, 3.0], "col_2": [11.0, 22.0, 33.0]})
        ss = StaticSamples(data)

        ss_copy = ss.copy()
        ss_copy.df.loc[0, "col_1"] = 12345.0

        assert id(ss_copy) != id(ss)
        assert id(ss_copy.df) != id(ss.df)
        assert id(ss_copy.df) != id(ss.df)
        assert ss.df.loc[0, "col_1"] == 1.0
        assert ss_copy.df.loc[0, "col_1"] == 12345.0

    class TestMutation:
        def test_mutate_df_loc(self):
            data = pd.DataFrame({"col_1": [1.0, 2.0, 3.0], "col_2": [11.0, 22.0, 33.0]})
            ss = StaticSamples(data)

            ss.df.loc[0, "col_1"] = 999.0

            assert ss.df.loc[0, "col_1"] == 999.0
            assert ss._data.loc[0, "col_1"] == 999.0  # pylint: disable=protected-access

        def test_mutate_df_slice_all(self):
            data = pd.DataFrame({"col_1": [1.0, 2.0, 3.0], "col_2": [11.0, 22.0, 33.0]})
            ss = StaticSamples(data)

            ss.df[:] = np.asarray([[9.0, 8.0, 6.0], [-1.0, -2.0, -3.0]]).T

            assert (ss.df == pd.DataFrame({"col_1": [9.0, 8.0, 6.0], "col_2": [-1.0, -2.0, -3.0]})).all().all()
            temp = ss._data  # pylint: disable=protected-access
            assert (temp == pd.DataFrame({"col_1": [9.0, 8.0, 6.0], "col_2": [-1.0, -2.0, -3.0]})).all().all()

        def test_mutate_df_reassign(self):
            data = pd.DataFrame({"col_1": [1.0, 2.0, 3.0], "col_2": [11.0, 22.0, 33.0]})
            ss = StaticSamples(data)

            ss.df = pd.DataFrame({"col_1": [-1.0, -2.0, -3.0], "col_2": [-11.0, -22.0, -33.0]})

            assert (ss.df == pd.DataFrame({"col_1": [-1.0, -2.0, -3.0], "col_2": [-11.0, -22.0, -33.0]})).all().all()
            temp = ss._data  # pylint: disable=protected-access
            assert (temp == pd.DataFrame({"col_1": [-1.0, -2.0, -3.0], "col_2": [-11.0, -22.0, -33.0]})).all().all()
