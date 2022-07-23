import os

import pytest

from clairvoyance2.datasets.download import download_file


class TestIntegration:
    @pytest.mark.internet
    def test_successful_download(self, tmpdir):
        test_url = "http://www.google.com/"
        test_file_path = os.path.join(tmpdir, "dummy_file")
        download_file(test_url, test_file_path)
        assert os.path.exists(test_file_path)
        assert os.path.isfile(test_file_path)
        assert os.path.getsize(test_file_path) > 0

    @pytest.mark.internet
    @pytest.mark.slow
    def test_successful_download_from_uci(self, tmpdir):
        # To notice if UCI's URLs have changed etc.
        test_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/diabetes/Index"
        test_file_path = os.path.join(tmpdir, "Index")
        download_file(test_url, test_file_path)
        assert os.path.exists(test_file_path)
        assert os.path.isfile(test_file_path)
        assert os.path.getsize(test_file_path) > 0
