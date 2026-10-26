import pytest

import json

from scripts.build_crop_shift_strata_reference import interpolated_quantile, load_frozen_counts


def test_interpolated_quantile_matches_linear_definition() -> None:
    values = [4.0, 1.0, 3.0, 2.0]
    assert interpolated_quantile(values, 0.0) == 1.0
    assert interpolated_quantile(values, 0.5) == 2.5
    assert interpolated_quantile(values, 1.0) == 4.0
    with pytest.raises(ValueError):
        interpolated_quantile([], 0.5)
    with pytest.raises(ValueError):
        interpolated_quantile(values, 1.1)


def test_load_frozen_counts_validates_vocab_alignment(tmp_path) -> None:
    path = tmp_path / "counts.json"
    path.write_text(json.dumps({"counts": [2, 0, 4]}), encoding="utf-8")
    counts, digest = load_frozen_counts(path, 3)
    assert counts == [2, 0, 4]
    assert len(digest) == 64
    with pytest.raises(ValueError, match="vocabulary size"):
        load_frozen_counts(path, 2)

