"""Reporting and verdict explanation package for Neurasym."""

from src.reporting.explain_verdict import (
    LABEL_COLORS,
    LABEL_LEGEND,
    LABEL_TOOLTIPS,
    explain_run,
    format_plain_english_box,
    format_plain_english_summary,
)

__all__ = [
    "explain_run",
    "LABEL_COLORS",
    "LABEL_TOOLTIPS",
    "LABEL_LEGEND",
    "format_plain_english_box",
    "format_plain_english_summary",
]
