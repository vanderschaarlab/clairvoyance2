import pytest

from clairvoyance2.datasets.uci import TimeSeriesSamples, UCIDiabetesRetriever

# pylint: disable=redefined-outer-name
# ^ Otherwise pylint trips up on pytest fixtures.

# pylint: disable=unused-argument
# ^ Some fixtures intentionally included without being explicitly used in test functions.


@pytest.fixture
def mock_dataset_root_dir(tmpdir, monkeypatch):
    monkeypatch.setattr(
        "clairvoyance2.datasets.dataset.DATASET_ROOT_DIR",
        tmpdir,
        raising=True,
    )


class TestIntegration:
    @pytest.mark.internet
    @pytest.mark.slow
    def test_uci_diabetes_retrieve_works(self, mock_dataset_root_dir):
        uci = UCIDiabetesRetriever()
        tss = uci.retrieve()

        assert isinstance(tss, TimeSeriesSamples)
        assert len(tss) == 70
