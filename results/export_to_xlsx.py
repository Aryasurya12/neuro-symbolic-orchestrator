"""Pure-Python XLSX Exporter for Neurasym Comparative Benchmark Results.

Generates 100% standard-compliant, beautifully styled Microsoft Excel (.xlsx) workbooks
without external dependencies (pure Python standard library: zipfile + xml).
"""

from __future__ import annotations

import csv
import html
import json
import os
import re
import sys
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


def escape_xml(val: Any) -> str:
    """Safely escapes text for XML."""
    if val is None:
        return ""
    text = str(val)
    # Remove ASCII control chars except tab, newline, cr
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", text)
    return html.escape(text, quote=True)


def col_letter(col_idx: int) -> str:
    """Converts 0-based column index to Excel column letters (e.g. 0 -> 'A', 27 -> 'AB')."""
    result = ""
    col_idx += 1
    while col_idx > 0:
        col_idx, remainder = divmod(col_idx - 1, 26)
        result = chr(65 + remainder) + result
    return result


class MinimalXlsxWorkbook:
    """Constructs a multi-sheet, styled Excel workbook using pure Python."""

    def __init__(self):
        self.sheets: List[Dict[str, Any]] = []
        self.shared_strings: List[str] = []
        self.string_to_idx: Dict[str, int] = {}

    def get_string_id(self, s: str) -> int:
        if s in self.string_to_idx:
            return self.string_to_idx[s]
        idx = len(self.shared_strings)
        self.shared_strings.append(s)
        self.string_to_idx[s] = idx
        return idx

    def add_sheet(self, title: str, rows: List[List[Tuple[Any, int]]], col_widths: Optional[List[float]] = None) -> None:
        """Adds a sheet with styled cell tuples: (value, style_idx)."""
        self.sheets.append({
            "title": title,
            "rows": rows,
            "col_widths": col_widths or [],
        })

    def build_content_types_xml(self) -> str:
        overrides = []
        for idx in range(1, len(self.sheets) + 1):
            overrides.append(
                f'<Override PartName="/xl/worksheets/sheet{idx}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            )
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">\n'
            '  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>\n'
            '  <Default Extension="xml" ContentType="application/xml"/>\n'
            '  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>\n'
            '  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>\n'
            '  <Override PartName="/xl/sharedStrings.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStringTable+xml"/>\n'
            '  ' + "\n  ".join(overrides) + "\n"
            '</Types>'
        )

    def build_root_rels_xml(self) -> str:
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
            '  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>\n'
            '</Relationships>'
        )

    def build_workbook_rels_xml(self) -> str:
        rels = []
        r_id = 1
        for idx in range(1, len(self.sheets) + 1):
            rels.append(
                f'<Relationship Id="rId{r_id}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{idx}.xml"/>'
            )
            r_id += 1
        rels.append(
            f'<Relationship Id="rId{r_id}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
        )
        r_id += 1
        rels.append(
            f'<Relationship Id="rId{r_id}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings" Target="sharedStrings.xml"/>'
        )
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
            '  ' + "\n  ".join(rels) + "\n"
            '</Relationships>'
        )

    def build_workbook_xml(self) -> str:
        sheets_xml = []
        for idx, s in enumerate(self.sheets, 1):
            title = escape_xml(s["title"])
            sheets_xml.append(f'<sheet name="{title}" sheetId="{idx}" r:id="rId{idx}"/>')
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">\n'
            '  <sheets>\n'
            '    ' + "\n    ".join(sheets_xml) + "\n"
            '  </sheets>\n'
            '</workbook>'
        )

    def build_shared_strings_xml(self) -> str:
        items = []
        for s in self.shared_strings:
            esc = escape_xml(s)
            items.append(f'<si><t xml:space="preserve">{esc}</t></si>')
        count = len(self.shared_strings)
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            f'<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" count="{count}" uniqueCount="{count}">\n'
            '  ' + "\n  ".join(items) + "\n"
            '</sst>'
        )

    def build_styles_xml(self) -> str:
        """Defines modern, professional palette of fonts, fills, borders, and style indexes.

        Style Map:
          0: Default cell (Regular text, left aligned)
          1: Main Table Header (Navy #1B365D, Bold White, Centered)
          2: Section Header (Teal #0B5345, Bold White, Left aligned)
          3: PASS cell (Green #D4EFDF fill, Dark Green #145A32 text, Bold, Centered)
          4: FAIL cell (Red #FADBD8 fill, Dark Red #78281F text, Bold, Centered)
          5: Number / Latency cell (Right aligned, monospace format)
          6: Zebra Row cell (Subtle #F8F9F9 fill, Left aligned)
          7: Zebra Number cell (Subtle #F8F9F9 fill, Right aligned)
          8: Bold Key / Subtitle (Bold text, left aligned)
          9: KPI Metric Value (14pt Bold, Center aligned, #EAECEE fill)
          10: KPI Label (9pt Gray, Center aligned, #EAECEE fill)
        """
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">\n'
            '  <fonts count="6">\n'
            '    <font><sz val="10"/><name val="Segoe UI"/><color rgb="FF2C3E50"/></font>\n'  # 0: Default
            '    <font><b/><sz val="11"/><name val="Segoe UI"/><color rgb="FFFFFFFF"/></font>\n'  # 1: Header White Bold
            '    <font><b/><sz val="10"/><name val="Segoe UI"/><color rgb="FF145A32"/></font>\n'  # 2: Pass Green Bold
            '    <font><b/><sz val="10"/><name val="Segoe UI"/><color rgb="FF78281F"/></font>\n'  # 3: Fail Red Bold
            '    <font><b/><sz val="10"/><name val="Segoe UI"/><color rgb="FF1B365D"/></font>\n'  # 4: Navy Bold
            '    <font><b/><sz val="14"/><name val="Segoe UI"/><color rgb="FF1B365D"/></font>\n'  # 5: Big KPI
            '  </fonts>\n'
            '  <fills count="7">\n'
            '    <fill><patternFill patternType="none"/></fill>\n'  # 0: None
            '    <fill><patternFill patternType="gray125"/></fill>\n'  # 1: Gray125
            '    <fill><patternFill patternType="solid"><fgColor rgb="FF1B365D"/></patternFill></fill>\n'  # 2: Navy Header
            '    <fill><patternFill patternType="solid"><fgColor rgb="FF0E6655"/></patternFill></fill>\n'  # 3: Teal Section
            '    <fill><patternFill patternType="solid"><fgColor rgb="FFD4EFDF"/></patternFill></fill>\n'  # 4: Light Green
            '    <fill><patternFill patternType="solid"><fgColor rgb="FFFADBD8"/></patternFill></fill>\n'  # 5: Light Red
            '    <fill><patternFill patternType="solid"><fgColor rgb="FFF8F9FA"/></patternFill></fill>\n'  # 6: Zebra Gray
            '  </fills>\n'
            '  <borders count="2">\n'
            '    <border><left/><right/><top/><bottom/><diagonal/></border>\n'  # 0: None
            '    <border>\n'  # 1: Thin Gray Gridlines
            '      <left style="thin"><color rgb="FFD5D8DC"/></left>\n'
            '      <right style="thin"><color rgb="FFD5D8DC"/></right>\n'
            '      <top style="thin"><color rgb="FFD5D8DC"/></top>\n'
            '      <bottom style="thin"><color rgb="FFD5D8DC"/></bottom>\n'
            '    </border>\n'
            '  </borders>\n'
            '  <cellStyleXfs count="1">\n'
            '    <xf numFmtId="0" fontId="0" fillId="0" borderId="0"/>\n'
            '  </cellStyleXfs>\n'
            '  <cellXfs count="11">\n'
            '    <xf numFmtId="0" fontId="0" fillId="0" borderId="1" xfId="0" applyFont="1" applyBorder="1" applyAlignment="1"><alignment vertical="center"/></xf>\n'  # 0: Default
            '    <xf numFmtId="0" fontId="1" fillId="2" borderId="1" xfId="0" applyFont="1" applyFill="1" applyBorder="1" applyAlignment="1"><alignment horizontal="center" vertical="center" wrapText="1"/></xf>\n'  # 1: Navy Header
            '    <xf numFmtId="0" fontId="1" fillId="3" borderId="1" xfId="0" applyFont="1" applyFill="1" applyBorder="1" applyAlignment="1"><alignment horizontal="left" vertical="center"/></xf>\n'  # 2: Teal Section
            '    <xf numFmtId="0" fontId="2" fillId="4" borderId="1" xfId="0" applyFont="1" applyFill="1" applyBorder="1" applyAlignment="1"><alignment horizontal="center" vertical="center"/></xf>\n'  # 3: Pass Green
            '    <xf numFmtId="0" fontId="3" fillId="5" borderId="1" xfId="0" applyFont="1" applyFill="1" applyBorder="1" applyAlignment="1"><alignment horizontal="center" vertical="center"/></xf>\n'  # 4: Fail Red
            '    <xf numFmtId="0" fontId="0" fillId="0" borderId="1" xfId="0" applyFont="1" applyBorder="1" applyAlignment="1"><alignment horizontal="right" vertical="center"/></xf>\n'  # 5: Number Right
            '    <xf numFmtId="0" fontId="0" fillId="6" borderId="1" xfId="0" applyFont="1" applyFill="1" applyBorder="1" applyAlignment="1"><alignment vertical="center"/></xf>\n'  # 6: Zebra Left
            '    <xf numFmtId="0" fontId="0" fillId="6" borderId="1" xfId="0" applyFont="1" applyFill="1" applyBorder="1" applyAlignment="1"><alignment horizontal="right" vertical="center"/></xf>\n'  # 7: Zebra Right
            '    <xf numFmtId="0" fontId="4" fillId="0" borderId="1" xfId="0" applyFont="1" applyBorder="1" applyAlignment="1"><alignment vertical="center"/></xf>\n'  # 8: Bold Text
            '    <xf numFmtId="0" fontId="5" fillId="6" borderId="1" xfId="0" applyFont="1" applyFill="1" applyBorder="1" applyAlignment="1"><alignment horizontal="center" vertical="center"/></xf>\n'  # 9: KPI Big
            '    <xf numFmtId="0" fontId="0" fillId="6" borderId="1" xfId="0" applyFont="1" applyFill="1" applyBorder="1" applyAlignment="1"><alignment horizontal="center" vertical="center"/></xf>\n'  # 10: KPI Label
            '  </cellXfs>\n'
            '</styleSheet>'
        )

    def build_worksheet_xml(self, sheet_dict: Dict[str, Any]) -> str:
        rows_data = sheet_dict["rows"]
        col_widths = sheet_dict.get("col_widths", [])

        # Column definitions
        cols_xml = []
        if col_widths:
            cols_xml.append("<cols>")
            for c_idx, w in enumerate(col_widths, 1):
                cols_xml.append(f'<col min="{c_idx}" max="{c_idx}" width="{w:.1f}" customWidth="1"/>')
            cols_xml.append("</cols>")

        sheet_data = []
        for r_idx, row_cells in enumerate(rows_data, 1):
            row_items = []
            for c_idx, cell_data in enumerate(row_cells):
                val, style_idx = cell_data
                cell_ref = f"{col_letter(c_idx)}{r_idx}"

                if val is None or val == "":
                    row_items.append(f'<c r="{cell_ref}" s="{style_idx}"/>')
                elif isinstance(val, (int, float)) and not isinstance(val, bool):
                    row_items.append(f'<c r="{cell_ref}" s="{style_idx}"><v>{val}</v></c>')
                else:
                    str_id = self.get_string_id(str(val))
                    row_items.append(f'<c r="{cell_ref}" s="{style_idx}" t="s"><v>{str_id}</v></c>')

            sheet_data.append(f'<row r="{r_idx}">{"".join(row_items)}</row>')

        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">\n'
            '  <sheetViews>\n'
            '    <sheetView tabSelected="1" workbookViewId="0">\n'
            '      <pane state="frozen" ySplit="1" topLeftCell="A2" activePane="bottomLeft"/>\n'
            '    </sheetView>\n'
            '  </sheetViews>\n'
            '  ' + "\n  ".join(cols_xml) + "\n"
            '  <sheetData>\n'
            '    ' + "\n    ".join(sheet_data) + "\n"
            '  </sheetData>\n'
            '</worksheet>'
        )

    def save(self, filepath: str) -> None:
        """Writes the complete zipped OpenXML .xlsx file to disk."""
        target_path = Path(filepath)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(target_path, "w", zipfile.ZIP_DEFLATED) as zf:
            # 1. Root content types
            zf.writestr("[Content_Types].xml", self.build_content_types_xml())

            # 2. Relationships
            zf.writestr("_rels/.rels", self.build_root_rels_xml())
            zf.writestr("xl/_rels/workbook.xml.rels", self.build_workbook_rels_xml())

            # 3. Workbook structure & styles
            zf.writestr("xl/workbook.xml", self.build_workbook_xml())
            zf.writestr("xl/styles.xml", self.build_styles_xml())

            # 4. Worksheets
            for idx, s in enumerate(self.sheets, 1):
                zf.writestr(f"xl/worksheets/sheet{idx}.xml", self.build_worksheet_xml(s))

            # 5. Shared string table (must be written last after all strings indexed)
            zf.writestr("xl/sharedStrings.xml", self.build_shared_strings_xml())


def generate_results_xlsx(
    jsonl_path: str = "results/live_exploration_log.jsonl",
    output_paths: Optional[List[str]] = None,
) -> int:
    """Reads JSONL logs and generates formatted .xlsx workbooks."""
    if output_paths is None:
        output_paths = ["results.xlsx", "results/results.xlsx"]

    records = []
    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass

    wb = MinimalXlsxWorkbook()

    # ---------------------------------------------------------
    # SHEET 1: All Mode Results (Granular Flat Table)
    # ---------------------------------------------------------
    headers = [
        "Timestamp (UTC)",
        "Natural Language Query",
        "Mode ID",
        "Execution Mode Paradigm",
        "Engine / Solver Provider",
        "Latency (ms)",
        "Reported Cost",
        "Mathematical Feasibility",
        "Verification Verdict",
        "Formal Proof Classification",
        "Stage 6 FinOps Explanation",
    ]
    sheet1_rows: List[List[Tuple[Any, int]]] = []
    # Header row
    sheet1_rows.append([(h, 1) for h in headers])

    for row_idx, rec in enumerate(records):
        ts = rec.get("timestamp", "")
        q = rec.get("query", "")
        res_dict = rec.get("results", {})
        zebra = (row_idx % 2 == 1)

        for m_key in ["mode1", "mode2", "mode3", "mode4"]:
            mdata = res_dict.get(m_key, {})
            if not mdata:
                continue

            feas = mdata.get("feasibility", "UNKNOWN")
            is_pass = "PASS" in feas
            feas_style = 3 if is_pass else 4

            lat_val = float(mdata.get("latency_ms", 0.0))
            base_style = 6 if zebra else 0
            num_style = 7 if zebra else 5

            sheet1_rows.append([
                (ts[:19].replace("T", " "), base_style),
                (q, base_style),
                (m_key.upper(), 8 if zebra else 8),
                (mdata.get("mode_name", m_key), base_style),
                (mdata.get("engine", "N/A"), base_style),
                (round(lat_val, 2), num_style),
                (str(mdata.get("reported_cost", "N/A")), num_style),
                (feas, feas_style),
                (mdata.get("verdict", "N/A"), base_style),
                (mdata.get("proof_verdict", "N/A"), base_style),
                (mdata.get("explanation", "N/A"), base_style),
            ])

    s1_widths = [20.0, 42.0, 10.0, 24.0, 32.0, 14.0, 14.0, 26.0, 32.0, 34.0, 48.0]
    wb.add_sheet("All Mode Results", sheet1_rows, s1_widths)

    # ---------------------------------------------------------
    # SHEET 2: Query Comparison Matrix
    # ---------------------------------------------------------
    s2_headers = [
        "Query ID",
        "User Query",
        "Mode 1 (Raw LLM)",
        "M1 Latency",
        "Mode 2 (Schema LLM)",
        "M2 Latency",
        "Mode 3 (Pure Symbolic)",
        "M3 Latency",
        "Mode 4 (Neuro-Symbolic)",
        "M4 Latency",
        "Architectural Takeaway",
    ]
    sheet2_rows: List[List[Tuple[Any, int]]] = []
    sheet2_rows.append([(h, 1) for h in s2_headers])

    for q_idx, rec in enumerate(records, 1):
        q = rec.get("query", "")
        res = rec.get("results", {})
        m1 = res.get("mode1", {})
        m2 = res.get("mode2", {})
        m3 = res.get("mode3", {})
        m4 = res.get("mode4", {})

        m1_f = m1.get("feasibility", "N/A")
        m2_f = m2.get("feasibility", "N/A")
        m3_f = m3.get("feasibility", "N/A")
        m4_f = m4.get("feasibility", "N/A")

        m1_st = 3 if "PASS" in m1_f else 4
        m2_st = 3 if "PASS" in m2_f else 4
        m3_st = 3 if "PASS" in m3_f else 4
        m4_st = 3 if "PASS" in m4_f else 4

        m3_pass = "PASS" in m3_f
        m4_pass = "PASS" in m4_f

        if m4_pass and m3_pass:
            takeaway = "Mode 4 (Neuro-Symbolic) Winner: Exact proof + executive report"
        elif (not m4_pass) and (not m3_pass):
            takeaway = "Constraint Overload Detected: Modes 3 & 4 safely rejected invalid configuration"
        else:
            takeaway = "Symbolic verification enforced"

        zebra = (q_idx % 2 == 1)
        b_st = 6 if zebra else 0
        n_st = 7 if zebra else 5

        sheet2_rows.append([
            (f"Q{q_idx}", 8),
            (q, b_st),
            (f"{m1_f} | {m1.get('reported_cost', '—')}", m1_st),
            (f"{float(m1.get('latency_ms', 0)):.1f}ms", n_st),
            (f"{m2_f} | {m2.get('reported_cost', '—')}", m2_st),
            (f"{float(m2.get('latency_ms', 0)):.1f}ms", n_st),
            (f"{m3_f} | {m3.get('proof_verdict', '—')}", m3_st),
            (f"{float(m3.get('latency_ms', 0)):.1f}ms", n_st),
            (f"{m4_f} | {m4.get('proof_verdict', '—')}", m4_st),
            (f"{float(m4.get('latency_ms', 0)):.1f}ms", n_st),
            (takeaway, b_st),
        ])

    s2_widths = [10.0, 42.0, 24.0, 12.0, 24.0, 12.0, 28.0, 12.0, 28.0, 12.0, 45.0]
    wb.add_sheet("Query Comparison Matrix", sheet2_rows, s2_widths)

    # ---------------------------------------------------------
    # SHEET 3: Executive Paradigm Summary & Stats
    # ---------------------------------------------------------
    sheet3_rows: List[List[Tuple[Any, int]]] = []
    sheet3_rows.append([("NEURASYM 4-WAY PARADIGM BENCHMARK EXECUTIVE SUMMARY", 2), ("", 2), ("", 2), ("", 2), ("", 2), ("", 2)])
    sheet3_rows.append([("Generated at: " + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"), 0), ("", 0), ("", 0), ("", 0), ("", 0), ("", 0)])
    sheet3_rows.append([("", 0), ("", 0), ("", 0), ("", 0), ("", 0), ("", 0)])

    # Paradigm Aggregate Stats
    sheet3_rows.append([("Paradigm Evaluation Mode", 1), ("Engine Description", 1), ("Queries Evaluated", 1), ("Feasible (PASS)", 1), ("Failed / Infeasible", 1), ("Avg Latency (ms)", 1)])

    for m_key, m_name, m_eng in [
        ("mode1", "Mode 1: Raw Unconstrained LLM", "Groq API (openai/gpt-oss-120b)"),
        ("mode2", "Mode 2: Schema-Constrained LLM", "Groq API (Strict Pydantic JSON)"),
        ("mode3", "Mode 3: Pure Symbolic Pipeline", "Local Solvers (HiGHS / PSO / Z3)"),
        ("mode4", "Mode 4: Full Neuro-Symbolic", "Local Solvers + NVIDIA Explainer"),
    ]:
        m_records = [r.get("results", {}).get(m_key, {}) for r in records if m_key in r.get("results", {})]
        total_q = len(m_records)
        pass_count = sum(1 for m in m_records if "PASS" in str(m.get("feasibility", "")))
        fail_count = total_q - pass_count
        avg_lat = (sum(float(m.get("latency_ms", 0.0)) for m in m_records) / total_q) if total_q > 0 else 0.0

        sheet3_rows.append([
            (m_name, 8),
            (m_eng, 0),
            (total_q, 5),
            (pass_count, 3 if pass_count > 0 else 0),
            (fail_count, 4 if fail_count > 0 else 0),
            (f"{avg_lat:.2f} ms", 5),
        ])

    sheet3_rows.append([("", 0), ("", 0), ("", 0), ("", 0), ("", 0), ("", 0)])
    sheet3_rows.append([("CORE ARCHITECTURAL TAKEAWAYS", 2), ("", 2), ("", 2), ("", 2), ("", 2), ("", 2)])
    sheet3_rows.append([("1. Mode 1 (Raw LLM): Fails mathematical feasibility due to ungrounded pricing & resource hallucinations.", 0), ("", 0), ("", 0), ("", 0), ("", 0), ("", 0)])
    sheet3_rows.append([("2. Mode 2 (Schema LLM): Conforms to JSON syntax, but suffers arithmetic deficits and violates catalog pricing constraints.", 0), ("", 0), ("", 0), ("", 0), ("", 0), ("", 0)])
    sheet3_rows.append([("3. Mode 3 (Pure Symbolic): Guarantees 100% deterministic mathematical bounds, but lacks executive explainability.", 0), ("", 0), ("", 0), ("", 0), ("", 0), ("", 0)])
    sheet3_rows.append([("4. Mode 4 (Neuro-Symbolic): Combines deterministic mathematical guarantees with professional NVIDIA-synthesized FinOps deployment reports.", 8), ("", 8), ("", 8), ("", 8), ("", 8), ("", 8)])

    s3_widths = [32.0, 34.0, 18.0, 16.0, 18.0, 18.0]
    wb.add_sheet("Executive Summary", sheet3_rows, s3_widths)

    # Save to all requested paths
    for p in output_paths:
        wb.save(p)

    return len(records) * 4


def main():
    jsonl_path = "results/live_exploration_log.jsonl"
    out_paths = ["results.xlsx", "results/results.xlsx"]
    eval_count = generate_results_xlsx(jsonl_path, out_paths)
    print(f"Successfully generated Excel workbook at:")
    for p in out_paths:
        print(f"  -> {os.path.abspath(p)} ({os.path.getsize(p):,} bytes)")


if __name__ == "__main__":
    main()
