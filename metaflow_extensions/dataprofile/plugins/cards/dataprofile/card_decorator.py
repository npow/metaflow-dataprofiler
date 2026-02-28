"""
metaflow-dataprofiler
=====================
Automatic DataFrame profiling card for Metaflow, powered by ydata-profiling.

Drop ``@card(type='dataprofile')`` onto any step to get a rich EDA report for
every ``pandas.DataFrame`` artifact — zero changes needed inside the step.
"""

import re
import traceback
from html import escape as _esc

from metaflow.cards import MetaflowCard, MetaflowCardComponent

_SAMPLE_DEFAULT = 50_000
_RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _require_ydata():
    try:
        import ydata_profiling  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "ydata-profiling is required for metaflow-dataprofiler.\n"
            "Install with:  pip install ydata-profiling"
        ) from exc


def _apply_sample(df, sample_size, name="DataFrame"):
    """Return (df_maybe_sampled, warning_html_or_None)."""
    if sample_size is None or len(df) <= sample_size:
        return df, None
    sampled = df.sample(n=sample_size, random_state=_RANDOM_SEED)
    warning = (
        f"&#9888;&#65039; Profiled a random sample of <b>{sample_size:,}</b> / "
        f"<b>{len(df):,}</b> rows for <code>{_esc(str(name))}</code>. "
        f"Set <code>options={{'sample': None}}</code> to profile all rows "
        f"(may be slow)."
    )
    return sampled, warning


def _profile_df(df, title, minimal, sample_size):
    """Return (full_html_str, [warning_html]) or raise."""
    _require_ydata()
    from ydata_profiling import ProfileReport

    df_s, w = _apply_sample(df, sample_size, title)
    warnings = [w] if w else []
    report = ProfileReport(df_s, minimal=minimal, title=title, explorative=not minimal)
    return report.to_html(), warnings


def _profile_comparison(df1, name1, df2, name2, sample_size):
    """Return (full_html_str, [warning_html]) or raise."""
    _require_ydata()
    from ydata_profiling import ProfileReport

    df1_s, w1 = _apply_sample(df1, sample_size, name1)
    df2_s, w2 = _apply_sample(df2, sample_size, name2)
    warnings = [w for w in [w1, w2] if w]
    r1 = ProfileReport(df1_s, minimal=True, title=name1)
    r2 = ProfileReport(df2_s, minimal=True, title=name2)
    html = r1.compare(r2).to_html()
    return html, warnings


def _inject_banner(html, messages):
    """Inject a warning banner <div> right after the opening <body> tag."""
    if not messages:
        return html
    items = "".join(f"<p style='margin:4px 0'>{m}</p>" for m in messages)
    banner = (
        "<div style='"
        "background:#fff8e1;border:1px solid #f9a825;border-radius:6px;"
        "padding:12px 16px;margin:12px;"
        "font-family:-apple-system,BlinkMacSystemFont,\"Segoe UI\",sans-serif;"
        "font-size:14px;"
        f"'>{items}</div>"
    )
    return html.replace("<body>", f"<body>{banner}", 1)


def _split_html(full_html):
    """Split a complete ydata-profiling HTML into (head_block, body_content).

    head_block : everything inside <head>…</head> (styles + scripts)
    body_content : everything inside <body>…</body>
    Falls back to returning ("", full_html) if the document can't be parsed.
    """
    head_m = re.search(r"<head[^>]*>(.*?)</head>", full_html, re.DOTALL | re.IGNORECASE)
    body_m = re.search(r"<body[^>]*>(.*?)</body>", full_html, re.DOTALL | re.IGNORECASE)
    if head_m and body_m:
        return head_m.group(1), body_m.group(1)
    return "", full_html


def _html_to_iframe(html_str, height="900px"):
    """Embed a full HTML document in an <iframe> via base64 data URI.

    Used only for DataProfileComponent (inline use in generic cards), where
    there is no multi-tab wrapper and a single iframe is fine.
    """
    import base64

    encoded = base64.b64encode(html_str.encode("utf-8")).decode("ascii")
    src = f"data:text/html;base64,{encoded}"
    return (
        f'<iframe src="{src}" '
        f'style="width:100%;height:{height};border:none;display:block;"></iframe>'
    )


def _empty_pane(name):
    return (
        "<div style='padding:40px;text-align:center;"
        "font-family:-apple-system,BlinkMacSystemFont,\"Segoe UI\",sans-serif;"
        "color:#888;'>"
        f"<h3>&#9888;&#65039; Empty DataFrame: <code>{_esc(name)}</code></h3>"
        "<p>This DataFrame has no rows or columns to profile.</p>"
        "</div>"
    )


def _error_pane(name, exc_text):
    return (
        "<div style='padding:20px;font-family:-apple-system,BlinkMacSystemFont,"
        "\"Segoe UI\",sans-serif;'>"
        f"<h3 style='color:#c62828;'>&#10060; Profiling error: "
        f"<code>{_esc(name)}</code></h3>"
        f"<pre style='background:#fce4ec;padding:12px;border-radius:4px;"
        f"overflow:auto;font-size:12px;'>{_esc(exc_text)}</pre></div>"
    )


def _wrap_full(inner_html):
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<style>body{margin:0;background:#f8f8f8}</style></head>"
        f"<body>{inner_html}</body></html>"
    )


_NAV_CSS = """\
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
     background:#f8f8f8}
.dp-nav{display:flex;flex-wrap:wrap;gap:4px;padding:8px 10px 0;
        background:#16213e;border-bottom:3px solid #e94560;
        position:sticky;top:0;z-index:9999}
.dp-btn{padding:7px 16px;border:none;border-radius:5px 5px 0 0;
        background:rgba(255,255,255,.12);color:#bbb;cursor:pointer;
        font-size:13px;font-weight:500;transition:all .15s}
.dp-btn:hover{background:rgba(255,255,255,.22);color:#fff}
.dp-btn.dp-a{background:#e94560;color:#fff}
.dp-section{display:none}
.dp-section.dp-a{display:block}
"""

_NAV_JS = """\
(function(){
  document.querySelectorAll('.dp-btn').forEach(function(b){
    b.addEventListener('click',function(){
      document.querySelectorAll('.dp-btn').forEach(function(x){x.classList.remove('dp-a')});
      document.querySelectorAll('.dp-section').forEach(function(x){x.classList.remove('dp-a')});
      b.classList.add('dp-a');
      document.getElementById('dp-s'+b.dataset.idx).classList.add('dp-a');
    });
  });
})();
"""


def _tabbed_html(panes):
    """
    panes: list of (tab_label: str, profile_html_or_placeholder: str, is_full_html: bool)

    Inlines all content directly — no nested iframes — so the card renders
    correctly inside the Metaflow UI's sandboxed card container regardless of
    iframe restrictions or container height limits.

    For full ydata-profiling HTML documents: extracts head + body and inlines
    them. Head resources (CSS/JS) are deduplicated — included only once from
    the first profiled DataFrame, since all reports share the same libraries.
    For placeholder panes (empty/error): the HTML fragment is used directly.
    """
    nav = ""
    sections = ""
    shared_head = None  # head block from the first real profile (shared libs)

    for i, (label, content, is_full_html) in enumerate(panes):
        active = " dp-a" if i == 0 else ""
        nav += f'<button class="dp-btn{active}" data-idx="{i}">{_esc(label)}</button>\n'

        if is_full_html:
            head_block, body_content = _split_html(content)
            if shared_head is None:
                shared_head = head_block
            sections += (
                f'<div class="dp-section{active}" id="dp-s{i}">'
                f'{body_content}'
                f'</div>\n'
            )
        else:
            # Placeholder (empty DF, error) — just a small HTML fragment
            sections += (
                f'<div class="dp-section{active}" id="dp-s{i}">'
                f'{content}'
                f'</div>\n'
            )

    head_html = f"<style>{_NAV_CSS}</style>"
    if shared_head:
        head_html += shared_head

    return (
        f"<!DOCTYPE html>\n<html><head><meta charset='utf-8'>"
        f"{head_html}</head>\n"
        f"<body>\n<div class='dp-nav'>\n{nav}</div>\n{sections}"
        f"<script>{_NAV_JS}</script>\n</body></html>"
    )


def _collect_dataframes(task):
    """Iterate a Metaflow Task; return {name: DataFrame} for DataFrame artifacts."""
    import pandas as pd

    dfs = {}
    for artifact in task:
        name = artifact.id
        if name.startswith("_"):
            continue
        try:
            value = artifact.data
            if isinstance(value, pd.DataFrame):
                dfs[name] = value
        except Exception:
            pass
    return dfs


# ---------------------------------------------------------------------------
# DataProfileComponent — for current.card.append() in any card
# ---------------------------------------------------------------------------


class DataProfileComponent(MetaflowCardComponent):
    """
    Embed a ydata-profiling report for a DataFrame inside *any* Metaflow card.

    Usage::

        from metaflow_dataprofiler import DataProfileComponent

        @card
        @step
        def clean(self):
            raw = load()
            current.card.append(DataProfileComponent(raw, title="Before cleaning"))
            self.df = clean(raw)
            current.card.append(DataProfileComponent(self.df, title="After cleaning"))

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to profile.
    title : str
        Section heading shown in the profile report.
    minimal : bool
        Skip correlations / interactions (faster). Default False.
    sample : int | None
        Row cap before profiling. Default 50,000. Set None for no cap.
    """

    REALTIME_UPDATABLE = False

    def __init__(self, df, title="DataFrame Profile", minimal=False, sample=_SAMPLE_DEFAULT):
        self._df = df
        self._title = title
        self._minimal = minimal
        self._sample = sample

    def render(self):
        if self._df is None or self._df.empty:
            return _empty_pane(self._title)
        try:
            html, warnings = _profile_df(self._df, self._title, self._minimal, self._sample)
            if warnings:
                html = _inject_banner(html, warnings)
            return _html_to_iframe(html)
        except Exception:
            return _error_pane(self._title, traceback.format_exc())


# ---------------------------------------------------------------------------
# DataProfileCard — @card(type='dataprofile')
# ---------------------------------------------------------------------------


class DataProfileCard(MetaflowCard):
    """
    Auto-profiles every ``pandas.DataFrame`` artifact on a step.

    Add to any step with::

        @card(type='dataprofile')
        @step
        def process(self):
            self.train_df = pd.read_parquet("train.parquet")
            self.test_df  = pd.read_parquet("test.parquet")

    The card discovers all DataFrame attributes after the step finishes,
    profiles each one, and renders a tabbed HTML report. No changes needed
    inside the step.

    Options
    -------
    include : list[str]
        Profile only these artifact names (default: all DataFrames).
    exclude : list[str]
        Skip these artifact names.
    compare : list[str, str]
        Two artifact names to diff side-by-side using ydata-profiling comparison.
    sample : int | None
        Row cap before profiling (default 50,000). ``None`` disables sampling.
    minimal : bool
        Skip correlations and interactions (faster). Default False.
    title : str
        Tab label override (only used when there is a single DataFrame).
    """

    type = "dataprofile"
    ALLOW_USER_COMPONENTS = True

    def __init__(self, options=None, components=None, graph=None, flow=None, **kwargs):
        self._options = options or {}
        self._components = components or []

    def render(self, task):
        opts = self._options
        include = opts.get("include")
        exclude = opts.get("exclude") or []
        compare = opts.get("compare")
        sample_size = opts.get("sample", _SAMPLE_DEFAULT)
        minimal = opts.get("minimal", False)
        title_override = opts.get("title")

        dfs = _collect_dataframes(task)

        # Apply include / exclude filters
        if include:
            dfs = {k: v for k, v in dfs.items() if k in include}
        if exclude:
            dfs = {k: v for k, v in dfs.items() if k not in exclude}

        if not dfs:
            return _wrap_full(
                "<div style='padding:40px;text-align:center;"
                "font-family:-apple-system,BlinkMacSystemFont,\"Segoe UI\",sans-serif;"
                "color:#888;'>"
                "<h2>No DataFrames Found</h2>"
                "<p>No <code>pandas.DataFrame</code> artifacts were found on this step.</p>"
                "<p>Assign DataFrames to <code>self.*</code> attributes inside the step.</p>"
                "</div>"
            )

        # ── comparison mode ────────────────────────────────────────────────
        if compare and len(compare) == 2:
            return self._render_compare(dfs, compare, sample_size)

        # ── single DataFrame ───────────────────────────────────────────────
        if len(dfs) == 1:
            name, df = next(iter(dfs.items()))
            label = title_override or name
            return self._render_one(df, name, label, minimal, sample_size)

        # ── multiple DataFrames → tabbed layout ────────────────────────────
        return self._render_multi(dfs, minimal, sample_size)

    # ── private rendering helpers ──────────────────────────────────────────

    def _render_one(self, df, artifact_name, label, minimal, sample_size):
        if df.empty:
            return _wrap_full(_empty_pane(artifact_name))
        try:
            html, warnings = _profile_df(df, label, minimal, sample_size)
            if warnings:
                html = _inject_banner(html, warnings)
            return html
        except Exception:
            return _wrap_full(_error_pane(artifact_name, traceback.format_exc()))

    def _render_multi(self, dfs, minimal, sample_size):
        panes = []
        for name, df in dfs.items():
            if df.empty:
                panes.append((name, _empty_pane(name), False))
                continue
            try:
                html, warnings = _profile_df(df, name, minimal, sample_size)
                if warnings:
                    html = _inject_banner(html, warnings)
                panes.append((name, html, True))
            except Exception:
                panes.append((name, _error_pane(name, traceback.format_exc()), False))
        return _tabbed_html(panes)

    def _render_compare(self, dfs, compare, sample_size):
        name1, name2 = compare
        df1, df2 = dfs.get(name1), dfs.get(name2)
        missing = [n for n in compare if n not in dfs]
        if missing:
            names_html = "".join(f"<code>{_esc(n)}</code> " for n in missing)
            return _wrap_full(
                "<div style='padding:40px;font-family:-apple-system,BlinkMacSystemFont,"
                "\"Segoe UI\",sans-serif;color:#c62828;'>"
                "<h2>Comparison Error</h2>"
                f"<p>DataFrames not found: {names_html}</p></div>"
            )
        try:
            html, warnings = _profile_comparison(df1, name1, df2, name2, sample_size)
            if warnings:
                html = _inject_banner(html, warnings)
            return html
        except Exception:
            return _wrap_full(
                _error_pane(f"{name1} vs {name2}", traceback.format_exc())
            )


CARDS = [DataProfileCard]
