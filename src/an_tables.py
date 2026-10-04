"""Table output: every table is written as CSV and as a Markdown file; numbers are rounded only on the way out."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import an_style as style

PCT_COLS = ("top1", "top3", "ci_lo", "ci_hi", "chance", "valid_rate")


def _fmt(v, nd=3):
    if isinstance(v, float):
        if pd.isna(v):
            return ""
        if abs(v) >= 1000:
            return f"{v:,.0f}"
        if v != 0 and abs(v) < 10 ** -nd:
            return f"{v:.1e}"
        return f"{v:.{nd}f}"
    return str(v)


def to_markdown(df: pd.DataFrame, nd: int = 3) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(_fmt(v, nd).replace("|", "/") for v in row) + " |")
    return "\n".join(lines) + "\n"


def write_table(df: pd.DataFrame, folder: Path, name: str, title: str, note: str = "", nd: int = 3) -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    name = style.PREFIX + name
    df.to_csv(folder / f"{name}.csv", index=False)
    return folder / f"{name}.csv"


def gee_wide(gee: pd.DataFrame) -> pd.DataFrame:
    """Manuscript layout of the pooled GEE table: one row per condition/spec/term with OR [95% CI] and p."""
    g = gee[gee.term.notna() & gee.OR.notna()].copy()
    g["OR [95% CI]"] = g.apply(lambda r: f"{r.OR:.2f} [{r.ci_lo:.2f}, {r.ci_hi:.2f}]", axis=1)
    g["p"] = g.p.map(lambda p: "<0.0001" if p < 1e-4 else f"{p:.4f}")
    return g[["condition", "spec", "term", "OR [95% CI]", "p", "n", "clusters"]]
