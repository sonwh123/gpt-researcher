"""Template loading utilities for the ``sub_template`` report type.

A template describes the desired table of contents of the final report. The
Planner decomposes it into per-section sub-queries and the Publisher writes the
report following its structure.

Two input formats are supported:

- ``.txt`` : a free-form outline (used verbatim). Example::

      Section 1: P&L highlights result
        Sub Section 1.1: Revenue results, QoQ/YoY changes with reasons
        Sub Section 1.2: Wafer sales and the breakdown to quantity and ASP
      Section 2: Segment or Platform highlights
        Sub Section 2.1: Sales by segment, margins, and management comments
        Sub Section 2.2: Sales guidance/forecast/trend by segment

- ``.json`` : a structured outline, normalized to the text form above::

      {
        "title": "2024 Q2 Financial Report",
        "sections": [
          {"heading": "P&L highlights result",
           "subsections": ["Revenue results, QoQ/YoY changes with reasons",
                           "Wafer sales and the breakdown to quantity and ASP"]},
          {"heading": "Segment or Platform highlights",
           "subsections": ["Sales by segment, margins, and management comments",
                           "Sales guidance/forecast/trend by segment"]}
        ]
      }

The whole pipeline (decomposition prompt and report-writing prompt) always
consumes the normalized *text* outline, so both formats behave identically.
"""

import json
import os
import re
from typing import Any


def normalize_template(data: Any) -> str:
    """Normalize a parsed JSON template into a plain-text outline.

    Args:
        data: The parsed JSON structure. Expected to be a dict with an optional
            ``title`` and a ``sections`` list, where each section is a dict with
            a ``heading`` and an optional ``subsections`` list. Plain strings and
            lists are tolerated and rendered best-effort.

    Returns:
        A plain-text outline using ``Section N`` / ``Sub Section N.M`` markers.
    """
    # Tolerate a bare string or a list of section strings.
    if isinstance(data, str):
        return data.strip()
    if isinstance(data, list):
        data = {"sections": data}
    if not isinstance(data, dict):
        return str(data)

    lines: list[str] = []
    title = data.get("title")
    if title:
        lines.append(f"Report Title: {title}")

    sections = data.get("sections", [])
    for i, section in enumerate(sections, start=1):
        if isinstance(section, str):
            lines.append(f"Section {i}: {section}")
            continue
        if not isinstance(section, dict):
            lines.append(f"Section {i}: {section}")
            continue

        heading = section.get("heading") or section.get("title") or section.get("name") or ""
        lines.append(f"Section {i}: {heading}")

        subsections = section.get("subsections") or section.get("sub_sections") or []
        for j, sub in enumerate(subsections, start=1):
            sub_text = sub if isinstance(sub, str) else (
                sub.get("heading") or sub.get("title") or str(sub)
            ) if isinstance(sub, dict) else str(sub)
            lines.append(f"  Sub Section {i}.{j}: {sub_text}")

    return "\n".join(lines).strip()


def load_template(path: str) -> str:
    """Load a report template from a ``.txt`` or ``.json`` file.

    Args:
        path: Path to the template file. ``.json`` is parsed and normalized to a
            text outline; any other extension is read as raw text.

    Returns:
        The normalized plain-text outline of the template.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If a ``.json`` file cannot be parsed.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Template file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    if path.lower().endswith(".json"):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON template at {path}: {e}") from e
        return normalize_template(data)

    return raw.strip()


_SECTION_RE = re.compile(r'^\s*Section\s+(\d+)\s*:\s*(.*)$', re.IGNORECASE)
_SUBSECTION_RE = re.compile(r'^\s*Sub[\s_-]?Section\s+(\d+)\.(\d+)\s*:\s*(.*)$', re.IGNORECASE)


def parse_template_outline(normalized_text: str) -> list[dict]:
    """Parse the normalized ``Section N: ...`` / ``  Sub Section N.M: ...``
    text outline (the same format both JSON- and TXT-origin templates are
    reduced to, per ``normalize_template``/``load_template`` above) into a
    nested structure, used by report_type "sub_template_isolated" to know
    the deterministic order and nesting of leaf headings.

    Grouping is based purely on line order (a "Section" line opens a new
    section; every "Sub Section" line up to the next "Section" line belongs
    to it) rather than cross-validating the "N.M" numeric prefix against the
    parent's "N" - this keeps parsing robust to numbering typos.

    Lines that match neither pattern (a title line, blank lines, free prose)
    are ignored for structure purposes; the raw template text itself is
    unaffected and still goes to prompts whole.

    Returns, in document order:
        [{"heading": "Section 1: ...", "subsections": [
            {"heading": "Sub Section 1.1: ..."},
            {"heading": "Sub Section 1.2: ..."},
        ]},
         {"heading": "Section 2: ...", "subsections": []}]

    Returns [] if no "Section N:" line is found at all.
    """
    outline: list[dict] = []
    current_section: dict | None = None

    for line in normalized_text.splitlines():
        section_match = _SECTION_RE.match(line)
        if section_match:
            current_section = {
                "heading": line.strip(),
                "subsections": [],
            }
            outline.append(current_section)
            continue

        subsection_match = _SUBSECTION_RE.match(line)
        if subsection_match and current_section is not None:
            current_section["subsections"].append({"heading": line.strip()})

    return outline


def get_leaf_nodes(outline: list[dict]) -> list[dict]:
    """Flatten ``parse_template_outline()``'s nested structure into the
    ordered list of independent research+write units ("leaves") used by
    report_type "sub_template_isolated".

    - A Section WITH subsections contributes each Sub Section as one leaf
      (marker "###"); the Section itself is NOT a leaf - it's rendered as a
      plain wrapper header with no independent content.
    - A Section WITH NO subsections is itself one leaf (marker "##").

    Each returned dict: {"heading": str, "marker": "##" | "###",
    "parent_heading": str | None}. Order matches template order exactly.
    """
    leaves: list[dict] = []
    for section in outline:
        if section["subsections"]:
            for sub in section["subsections"]:
                leaves.append({
                    "heading": sub["heading"],
                    "marker": "###",
                    "parent_heading": section["heading"],
                })
        else:
            leaves.append({
                "heading": section["heading"],
                "marker": "##",
                "parent_heading": None,
            })
    return leaves
