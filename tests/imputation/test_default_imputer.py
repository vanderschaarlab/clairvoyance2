import numpy as np
import pandas as pd

from clairvoyance2.data import Dataset, TimeSeriesSamples
from clairvoyance2.imputation import DefaultImputerTC


class TestIntegration:
    def test_imputation_success(self):
        df_0 = pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0],
                "b": [10.0, np.nan, np.nan],
                "c": [np.nan, 22.0, 33.0],
                "d": [np.nan, np.nan, np.nan],
            }
        )
        df_1 = pd.DataFrame(
            {
                "a": [-1.0, -2.0, -3.0],
                "b": [-10.0, np.nan, np.nan],
                "c": [np.nan, -22.0, -33.0],
                "d": [100.0, np.nan, np.nan],
            }
        )
        ds = Dataset(TimeSeriesSamples([df_0, df_1]))
        imputer = DefaultImputerTC()

        before_has_missing = ds.temporal_covariates.has_missing
        ds = imputer.fit_transform(ds)
        after_has_missing = ds.temporal_covariates.has_missing

        assert before_has_missing is True
        assert after_has_missing is False
