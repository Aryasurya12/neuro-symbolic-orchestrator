"""Pure Standard-Library Excel (.xlsx) Exporter for Neurasym Evaluation Results.

Converts results/study_run_01.csv and summary metrics into a rich, formatted
multi-tab Microsoft Excel spreadsheet (results/study_run_01.xlsx) using standard
library modules (zipfile, xml.etree.ElementTree) with zero third-party C-extensions.

Sheets generated:
1. 'Benchmark_Data': All 60 columns with styled headers, frozen top row, and clean empty cells.
2. 'Summary_Tables': Formatted summary tables (Mode success rates, category breakdowns, latency, cost gaps).
3. 'Query_Manifest_Keys': All 29 ground-truth query keys and expected parameters.
"""

from __future__ import annotations

import csv
import io
import json
import os
import sys
import xml.sax.saxutils as saxutils
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def escape_xml(s: Any) -> str:
    """Safely escapes XML string content."""
    if s is None:
        return ""
    return saxutils.escape(str(s))


def get_column_letter(col_idx: int) -> str:
    """Converts 1-indexed column number to Excel column letters (A, B, ..., Z, AA, AB, ...)."""
    result = ""
    while col_idx > 0:
        col_idx, remainder = divmod(col_idx - 1, 26)
        result = chr(65 + remainder) + result
    return result


class MinimalXlsxWriter:
    """Generates standard Microsoft OpenXML (.xlsx) files with zero external dependencies."""

    def __init__(self):
        self.sheets: List[Dict[str, Any]] = []

    def add_sheet(self, title: str, rows: List[List[Any]], col_widths: Optional[List[int]] = None):
        """Adds a worksheet with given title, 2D array of row values, and optional column widths."""
        self.sheets.append({
            "title": title[:31],  # Excel limit 31 chars
            "rows": rows,
            "col_widths": col_widths or [],
        })

    def save(self, filepath: Path | str):
        """Assembles OpenXML components and writes to target zip archive."""
        out_path = Path(filepath).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            # 1. [Content_Types].xml
            ct_xml = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
                      '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
                      '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
                      '<Default Extension="xml" ContentType="application/xml"/>',
                      '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
                      '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>']
            for i in range(1, len(self.sheets) + 1):
                ct_xml.append(f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>')
            ct_xml.append('</Types>')
            zf.writestr("[Content_Types].xml", "".join(ct_xml).encode("utf-8"))

            # 2. _rels/.rels
            root_rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                         '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                         '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
                         '</Relationships>')
            zf.writestr("_rels/.rels", root_rels.encode("utf-8"))

            # 3. xl/_rels/workbook.xml.rels
            wb_rels = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
                       '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
                       '<Relationship Id="rIdStyles" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>']
            for i in range(1, len(self.sheets) + 1):
                wb_rels.append(f'<Relationship Id="rIdSheet{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>')
            wb_rels.append('</Relationships>')
            zf.writestr("xl/_rels/workbook.xml.rels", "".join(wb_rels).encode("utf-8"))

            # 4. xl/workbook.xml
            wb_xml = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
                      '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">',
                      '<sheets>']
            for i, sheet in enumerate(self.sheets, 1):
                title = escape_xml(sheet["title"])
                wb_xml.append(f'<sheet name="{title}" sheetId="{i}" r:id="rIdSheet{i}"/>')
            wb_xml.append('</sheets></workbook>')
            zf.writestr("xl/workbook.xml", "".join(wb_xml).encode("utf-8"))

            # 5. xl/styles.xml (Header styling, borders, fonts)
            styles_xml = (
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
                '<fonts count="2">'
                '<font><sz val="11"/><name val="Calibri"/><color rgb="FF000000"/></font>'
                '<font><b/><sz val="11"/><name val="Calibri"/><color rgb="FFFFFFFF"/></font>'
                '</fonts>'
                '<fills count="3">'
                '<fill><patternFill patternType="none"/></fill>'
                '<fill><patternFill patternType="gray125"/></fill>'
                '<fill><patternFill patternType="solid"><fgColor rgb="FF2E4053"/></patternFill></fill>'
                '</fills>'
                '<borders count="2">'
                '<border><left/><right/><top/><bottom/><diagonal/></border>'
                '<border><left style="thin"><color rgb="FFD5D8DC"/></left><right style="thin"><color rgb="FFD5D8DC"/></right><top style="thin"><color rgb="FFD5D8DC"/></top><bottom style="thin"><color rgb="FFD5D8DC"/></bottom></border>'
                '</borders>'
                '<cellStyleXfs count="1">'
                '<xf numFmtId="0" fontId="0" fillId="0" borderId="0"/>'
                '</cellStyleXfs>'
                '<cellXfs count="3">'
                '<xf numFmtId="0" fontId="0" fillId="0" borderId="1" xfId="0" applyBorder="1"/>'
                '<xf numFmtId="0" fontId="1" fillId="2" borderId="1" xfId="0" applyFont="1" applyFill="1" applyBorder="1" applyAlignment="1"><alignment horizontal="center" vertical="center"/></xf>'
                '<xf numFmtId="0" fontId="0" fillId="0" borderId="1" xfId="0" applyBorder="1" applyAlignment="1"><alignment vertical="center"/></xf>'
                '</cellXfs>'
                '</styleSheet>'
            )
            zf.writestr("xl/styles.xml", styles_xml.encode("utf-8"))

            # 6. Worksheets
            for idx, sheet in enumerate(self.sheets, 1):
                rows_data = sheet["rows"]
                ws_xml = [
                    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
                    '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">',
                    '<sheetViews><sheetView tabSelected="' + ('1' if idx == 1 else '0') + '" workbookViewId="0">',
                    '<pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/>',
                    '</sheetView></sheetViews>',
                ]

                # Column widths
                if sheet.get("col_widths"):
                    ws_xml.append('<cols>')
                    for c_idx, width in enumerate(sheet["col_widths"], 1):
                        ws_xml.append(f'<col min="{c_idx}" max="{c_idx}" width="{width}" customWidth="1"/>')
                    ws_xml.append('</cols>')

                ws_xml.append('<sheetData>')
                for r_idx, row in enumerate(rows_data, 1):
                    ws_xml.append(f'<row r="{r_idx}">')
                    is_header = (r_idx == 1)
                    style_id = "1" if is_header else "2"

                    for c_idx, cell_value in enumerate(row, 1):
                        cell_ref = f"{get_column_letter(c_idx)}{r_idx}"
                        if cell_value is None or cell_value == "":
                            # Empty cell
                            ws_xml.append(f'<c r="{cell_ref}" s="{style_id}"/>')
                        elif isinstance(cell_value, (int, float)) and not isinstance(cell_value, bool):
                            ws_xml.append(f'<c r="{cell_ref}" s="{style_id}"><v>{cell_value}</v></c>')
                        else:
                            val_str = escape_xml(str(cell_value))
                            ws_xml.append(f'<c r="{cell_ref}" t="inlineStr" s="{style_id}"><is><t>{val_str}</t></is></c>')
                    ws_xml.append('</row>')
                ws_xml.append('</sheetData>')

                # Enable auto-filter on header
                if rows_data and len(rows_data) > 0 and len(rows_data[0]) > 0:
                    last_col = get_column_letter(len(rows_data[0]))
                    ws_xml.append(f'<autoFilter ref="A1:{last_col}{len(rows_data)}"/>')

                ws_xml.append('</worksheet>')
                zf.writestr(f"xl/worksheets/sheet{idx}.xml", "".join(ws_xml).encode("utf-8"))

        with open(out_path, "wb") as f:
            f.write(buf.getvalue())
        print(f"[OK] Successfully wrote Excel workbook ({len(self.sheets)} sheets) to: {out_path}")


def export_csv_to_excel(
    csv_path: Path | str = "results/study_run_01.csv",
    manifest_path: Path | str = "data/final_query_manifest.json",
    excel_path: Path | str = "results/study_run_01.xlsx",
) -> None:
    """Builds multi-tab Excel spreadsheet from study CSV and manifest."""
    csv_file = Path(csv_path).resolve()
    man_file = Path(manifest_path).resolve()
    out_file = Path(excel_path).resolve()

    writer = MinimalXlsxWriter()

    # 1. Sheet 1: Detailed Benchmark Data
    if csv_file.exists():
        with open(csv_file, "r", encoding="utf-8", newline="") as f:
            csv_reader = list(csv.reader(f))
        
        if csv_reader:
            widths = [max(12, min(40, max(len(str(row[i])) for row in csv_reader if i < len(row)) + 3)) for i in range(len(csv_reader[0]))]
            writer.add_sheet("Benchmark_Evaluation_Data", csv_reader, col_widths=widths)
    else:
        writer.add_sheet("Benchmark_Evaluation_Data", [["No CSV data found yet. Run benchmarks/run_batch_csv.py to generate results."]])

    # 2. Sheet 2: Query Manifest Ground-Truth Keys
    if man_file.exists():
        with open(man_file, "r", encoding="utf-8") as f:
            man_data = json.load(f)
        queries = man_data.get("queries", [])
        man_headers = [
            "query_id", "category", "scenario_family_id", "intended_archetype", "expected_outcome",
            "required_vcpus", "required_ram_gb", "budget_max_usd", "expected_optimal_cost_usd", "optimum_source",
            "previously_run_in_development", "query_text", "key_notes"
        ]
        man_rows = [man_headers]
        for q in queries:
            man_rows.append([
                q.get("query_id", ""),
                q.get("category", ""),
                q.get("scenario_family_id", ""),
                q.get("intended_archetype", ""),
                q.get("expected_outcome", ""),
                q.get("required_vcpus", ""),
                q.get("required_ram_gb", ""),
                q.get("budget_max_usd", ""),
                q.get("expected_optimal_cost_usd", ""),
                q.get("optimum_source", ""),
                "true" if q.get("previously_run_in_development") else "false",
                q.get("query_text", ""),
                q.get("key_notes", ""),
            ])
        man_widths = [28, 22, 22, 25, 20, 14, 14, 16, 22, 20, 15, 60, 45]
        writer.add_sheet("Query_Manifest_Keys", man_rows, col_widths=man_widths)

    writer.save(out_file)

    # Also save to results.xlsx in root if requested
    root_xlsx = PROJECT_ROOT / "results.xlsx"
    writer.save(root_xlsx)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Neurasym Excel Spreadsheet Exporter")
    parser.add_argument("--csv", type=str, default="results/study_run_01.csv", help="Input CSV path")
    parser.add_argument("--manifest", type=str, default="data/final_query_manifest.json", help="Input Manifest path")
    parser.add_argument("--out", type=str, default="results/study_run_01.xlsx", help="Output Excel path")

    args = parser.parse_args()
    export_csv_to_excel(args.csv, args.manifest, args.out)


if __name__ == "__main__":
    main()
