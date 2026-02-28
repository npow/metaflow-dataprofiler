"""
Unit tests for metaflow-card-dataprofile.

These tests run entirely without ydata-profiling by mocking the heavy call,
so they are fast and safe for CI.
"""

import importlib
import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ── import the card module directly ──────────────────────────────────────────
from metaflow_extensions.dataprofile.plugins.cards.dataprofile.card_decorator import (
    DataProfileCard,
    DataProfileComponent,
    _apply_sample,
    _collect_dataframes,
    _empty_pane,
    _error_pane,
    _html_to_iframe,
    _inject_banner,
    _tabbed_html,
    _wrap_full,
)
from metaflow_extensions.dataprofile.plugins.cards.dataprofile import CARDS


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def small_df():
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "age": rng.integers(18, 80, 50),
            "income": rng.exponential(40_000, 50),
            "city": rng.choice(["NYC", "LA", "SF"], 50),
            "score": rng.uniform(0, 1, 50),
            "nullcol": np.where(rng.random(50) > 0.5, rng.random(50), np.nan),
        }
    )


@pytest.fixture
def large_df():
    """A DF larger than the default sample threshold."""
    rng = np.random.default_rng(1)
    n = 60_000
    return pd.DataFrame({"x": rng.standard_normal(n), "y": rng.integers(0, 100, n)})


# Mock task helpers -----------------------------------------------------------


class _FakeArtifact:
    def __init__(self, name, data):
        self.id = name
        self._data = data

    @property
    def data(self):
        return self._data


class _BrokenArtifact:
    """Artifact whose .data property always raises — for testing error handling."""

    def __init__(self, name):
        self.id = name

    @property
    def data(self):
        raise RuntimeError("cannot load")


class _FakeTask:
    def __init__(self, **kwargs):
        self._artifacts = [_FakeArtifact(k, v) for k, v in kwargs.items()]

    def __iter__(self):
        return iter(self._artifacts)


# Fake HTML returned from "ydata-profiling"
_FAKE_HTML = "<html><head></head><body><p>profile</p></body></html>"


def _fake_profile_df(df, title, minimal, sample_size):
    return _FAKE_HTML, []


def _fake_profile_comparison(df1, n1, df2, n2, sample_size):
    return _FAKE_HTML, []


# ─────────────────────────────────────────────────────────────────────────────
# _apply_sample
# ─────────────────────────────────────────────────────────────────────────────


class TestApplySample:
    def test_no_sample_when_below_limit(self, small_df):
        result, warning = _apply_sample(small_df, 10_000)
        assert result is small_df  # same object, no copy
        assert warning is None

    def test_no_sample_when_limit_is_none(self, large_df):
        result, warning = _apply_sample(large_df, None)
        assert len(result) == len(large_df)
        assert warning is None

    def test_samples_when_above_limit(self, large_df):
        result, warning = _apply_sample(large_df, 1_000, name="my_df")
        assert len(result) == 1_000
        assert warning is not None
        assert "60,000" in warning
        assert "1,000" in warning
        assert "my_df" in warning

    def test_reproducible_sample(self, large_df):
        r1, _ = _apply_sample(large_df, 500)
        r2, _ = _apply_sample(large_df, 500)
        pd.testing.assert_frame_equal(r1, r2)


# ─────────────────────────────────────────────────────────────────────────────
# HTML helpers
# ─────────────────────────────────────────────────────────────────────────────


class TestHtmlHelpers:
    def test_inject_banner_inserts_after_body(self):
        html = "<html><body><p>content</p></body></html>"
        out = _inject_banner(html, ["Warning text"])
        assert out.index("<div") < out.index("<p>content")
        assert "Warning text" in out

    def test_inject_banner_no_op_when_empty(self):
        html = "<html><body></body></html>"
        assert _inject_banner(html, []) == html

    def test_html_to_iframe_wraps_data_uri(self):
        import base64
        html = '<html><body><p class="x">a &amp; b</p></body></html>'
        out = _html_to_iframe(html)
        assert out.startswith("<iframe")
        assert 'src="data:text/html;base64,' in out
        # Verify the base64 content round-trips back to the original HTML
        b64 = out.split('src="data:text/html;base64,')[1].split('"')[0]
        assert base64.b64decode(b64).decode("utf-8") == html

    def test_empty_pane_contains_name(self):
        out = _empty_pane("my_dataframe")
        assert "my_dataframe" in out
        assert "Empty" in out

    def test_error_pane_contains_name_and_exc(self):
        out = _error_pane("bad_df", "ValueError: oops")
        assert "bad_df" in out
        assert "ValueError" in out

    def test_tabbed_html_produces_all_tabs(self):
        # (label, content, is_full_html)
        panes = [
            ("alpha", "<p>A</p>", False),
            ("beta", "<p>B</p>", False),
            ("gamma", "<p>C</p>", False),
        ]
        out = _tabbed_html(panes)
        for label, _, _ in panes:
            assert label in out
        # First tab should have dp-a class
        assert "dp-a" in out
        # Should have 3 section divs
        assert out.count('class="dp-section') == 3

    def test_wrap_full_is_valid_html(self):
        out = _wrap_full("<p>hello</p>")
        assert out.startswith("<!DOCTYPE html>")
        assert "<p>hello</p>" in out


# ─────────────────────────────────────────────────────────────────────────────
# _collect_dataframes
# ─────────────────────────────────────────────────────────────────────────────


class TestCollectDataframes:
    def test_collects_dataframes(self, small_df):
        task = _FakeTask(my_df=small_df, other="a string", count=42)
        result = _collect_dataframes(task)
        assert set(result.keys()) == {"my_df"}
        pd.testing.assert_frame_equal(result["my_df"], small_df)

    def test_skips_underscore_artifacts(self, small_df):
        task = _FakeTask(_task_ok=True, my_df=small_df)
        result = _collect_dataframes(task)
        assert "_task_ok" not in result

    def test_multiple_dataframes(self, small_df):
        df2 = small_df.head(10).copy()
        task = _FakeTask(train=small_df, test=df2, meta="info")
        result = _collect_dataframes(task)
        assert set(result.keys()) == {"train", "test"}

    def test_silently_skips_unloadable_artifact(self, small_df):
        task = _FakeTask(good=small_df)
        task._artifacts.append(_BrokenArtifact("broken"))
        result = _collect_dataframes(task)
        # Should still return good artifact, silently skip broken one
        assert "good" in result
        assert "broken" not in result


# ─────────────────────────────────────────────────────────────────────────────
# CARDS registration
# ─────────────────────────────────────────────────────────────────────────────


def test_cards_list_registered():
    assert DataProfileCard in CARDS
    assert DataProfileCard.type == "dataprofile"


# ─────────────────────────────────────────────────────────────────────────────
# DataProfileCard.render — with _profile_df mocked
# ─────────────────────────────────────────────────────────────────────────────

_MOD = "metaflow_extensions.dataprofile.plugins.cards.dataprofile.card_decorator"


class TestDataProfileCardRender:
    def test_no_dataframes_returns_placeholder(self):
        card = DataProfileCard()
        task = _FakeTask(scalar=42, name="alice")
        html = card.render(task)
        assert "No DataFrames Found" in html

    def test_single_df_returns_profile(self, small_df):
        card = DataProfileCard()
        task = _FakeTask(my_df=small_df)
        with patch(f"{_MOD}._profile_df", return_value=(_FAKE_HTML, [])):
            html = card.render(task)
        assert "profile" in html

    def test_multiple_dfs_returns_tabs(self, small_df):
        card = DataProfileCard()
        task = _FakeTask(train=small_df, test=small_df.head(20))
        with patch(f"{_MOD}._profile_df", return_value=(_FAKE_HTML, [])):
            html = card.render(task)
        assert "dp-nav" in html
        assert "train" in html
        assert "test" in html

    def test_include_filter(self, small_df):
        card = DataProfileCard(options={"include": ["train"]})
        task = _FakeTask(train=small_df, test=small_df.head(10))
        with patch(f"{_MOD}._profile_df", return_value=(_FAKE_HTML, [])) as mock_pd:
            card.render(task)
        # _profile_df should only be called once (for train)
        assert mock_pd.call_count == 1

    def test_exclude_filter(self, small_df):
        card = DataProfileCard(options={"exclude": ["test"]})
        task = _FakeTask(train=small_df, test=small_df.head(10))
        with patch(f"{_MOD}._profile_df", return_value=(_FAKE_HTML, [])) as mock_pd:
            card.render(task)
        assert mock_pd.call_count == 1

    def test_compare_mode(self, small_df):
        card = DataProfileCard(options={"compare": ["train", "test"]})
        task = _FakeTask(train=small_df, test=small_df.head(20))
        with patch(f"{_MOD}._profile_comparison", return_value=(_FAKE_HTML, [])):
            html = card.render(task)
        assert "profile" in html

    def test_compare_missing_df_returns_error(self, small_df):
        card = DataProfileCard(options={"compare": ["train", "missing_df"]})
        task = _FakeTask(train=small_df)
        html = card.render(task)
        assert "missing_df" in html
        assert "not found" in html.lower() or "Error" in html

    def test_empty_df_renders_placeholder(self):
        card = DataProfileCard()
        empty = pd.DataFrame()
        task = _FakeTask(empty_df=empty)
        html = card.render(task)
        assert "Empty" in html or "empty" in html

    def test_profiling_error_renders_gracefully(self, small_df):
        card = DataProfileCard()
        task = _FakeTask(bad_df=small_df)
        with patch(f"{_MOD}._profile_df", side_effect=RuntimeError("boom")):
            html = card.render(task)
        assert "boom" in html or "error" in html.lower()

    def test_sampling_warning_injected_in_single_df(self, large_df):
        card = DataProfileCard(options={"sample": 100})
        task = _FakeTask(big=large_df)
        warning_html = f"sample of 100 / {len(large_df):,}"
        with patch(f"{_MOD}._profile_df", return_value=(_FAKE_HTML, [warning_html])):
            html = card.render(task)
        assert "sample" in html.lower()

    def test_title_override_single_df(self, small_df):
        card = DataProfileCard(options={"title": "My Custom Title"})
        task = _FakeTask(df=small_df)
        with patch(f"{_MOD}._profile_df", return_value=(_FAKE_HTML, [])) as mock_pd:
            html = card.render(task)
        # _profile_df should have been called with our custom title
        assert mock_pd.called, "expected _profile_df to be called"
        _, kwargs = mock_pd.call_args
        # title is the second positional arg: _profile_df(df, title, minimal, sample_size)
        title_arg = mock_pd.call_args[0][1]
        assert title_arg == "My Custom Title"


# ─────────────────────────────────────────────────────────────────────────────
# DataProfileComponent
# ─────────────────────────────────────────────────────────────────────────────


class TestDataProfileComponent:
    def test_empty_df_returns_placeholder(self):
        comp = DataProfileComponent(pd.DataFrame(), title="empty")
        out = comp.render()
        assert "Empty" in out

    def test_none_df_returns_placeholder(self):
        comp = DataProfileComponent(None, title="none")
        out = comp.render()
        assert "Empty" in out

    def test_normal_df_returns_iframe(self, small_df):
        comp = DataProfileComponent(small_df, title="Test")
        with patch(f"{_MOD}._profile_df", return_value=(_FAKE_HTML, [])):
            out = comp.render()
        assert "<iframe" in out

    def test_error_renders_gracefully(self, small_df):
        comp = DataProfileComponent(small_df, title="err")
        with patch(f"{_MOD}._profile_df", side_effect=RuntimeError("explode")):
            out = comp.render()
        assert "explode" in out


# ─────────────────────────────────────────────────────────────────────────────
# Public import from metaflow_dataprofiler
# ─────────────────────────────────────────────────────────────────────────────


def test_public_import():
    from metaflow_dataprofiler import DataProfileComponent as DPC

    assert DPC is DataProfileComponent
