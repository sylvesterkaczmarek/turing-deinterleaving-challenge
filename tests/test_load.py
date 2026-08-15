from pathlib import Path
from unittest import mock

from turing_deinterleaving_challenge.data import load


def test_val_subset_uses_hugging_face_directory_name(tmp_path: Path) -> None:
    with mock.patch.object(
        load, "snapshot_download", return_value=str(tmp_path)
    ) as snapshot_download:
        result = load.download_dataset(
            save_dir=tmp_path,
            subsets="val",
            modes=["stare", "scan"],
            hf_token="token",
        )

    assert result == str(tmp_path)
    assert sorted(snapshot_download.call_args.kwargs["allow_patterns"]) == [
        "scan/val_scan/*.h5",
        "stare/val_stare/*.h5",
    ]
