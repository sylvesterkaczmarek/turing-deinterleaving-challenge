import pytest

from turing_deinterleaving_challenge.data import dataset as dataset_module


def test_invalid_subset_is_rejected_before_download(monkeypatch):
    def unexpected_download(*args, **kwargs):
        raise AssertionError("download_dataset should not be called")

    monkeypatch.setattr(dataset_module, "download_dataset", unexpected_download)

    with pytest.raises(ValueError, match="Invalid subset"):
        dataset_module.DeinterleavingChallengeDataset(subset="invalid")
