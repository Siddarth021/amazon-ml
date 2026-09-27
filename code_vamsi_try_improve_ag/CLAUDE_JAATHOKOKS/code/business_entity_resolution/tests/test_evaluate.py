import polars as pl
import pytest

from er.evaluate import score


def _s1(n):
    return pl.DataFrame({"s1_idx": list(range(n)), "country": ["US"] * n}, schema={"s1_idx": pl.Int32, "country": pl.Utf8})


def _pairs(rows):
    return pl.DataFrame(rows, schema={"s1_idx": pl.Int32, "rec_idx": pl.Int32}, orient="row")


def test_readme_example():
    # pred [S2-00047, S2-00193, S3-00812] vs truth [S2-00047, S3-00812]
    r = score(_s1(1), _pairs([(0, 47), (0, 812)]), _pairs([(0, 47), (0, 193), (0, 812)]))
    assert r["f05"] == pytest.approx(0.7142857, abs=1e-6)


def test_singleton_empty_is_one():
    assert score(_s1(1), _pairs([]), _pairs([]))["f05"] == 1.0


def test_singleton_with_pred_is_zero():
    assert score(_s1(1), _pairs([]), _pairs([(0, 5)]))["f05"] == 0.0


def test_missed_is_zero():
    assert score(_s1(1), _pairs([(0, 5)]), _pairs([]))["f05"] == 0.0


def test_macro_over_all_s1():
    # S1 0 perfect, S1 1 singleton predicted empty (1.0), S1 2 fully missed (0.0)
    r = score(_s1(3), _pairs([(0, 1), (2, 3)]), _pairs([(0, 1)]))
    assert r["f05"] == pytest.approx(2 / 3)
    assert r["loss"]["fully_missed"] == pytest.approx(1 / 3)
