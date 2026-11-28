"""Render the submission Markdown to a print-ready PDF.

Kept in the repo so the document can be regenerated exactly, rather than
being a PDF that only exists on someone's laptop.

Run: python tools/build_submission_pdf.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import markdown
from weasyprint import HTML

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "docs" / "submission" / "LLM_FoundIt.md"
OUTPUT = ROOT / "docs" / "submission" / "LLM_FoundIt.pdf"

CSS = """
@page {
  size: A4;
  margin: 20mm 18mm 22mm 18mm;
  @bottom-center {
    content: counter(page) " / " counter(pages);
    font-family: Georgia, serif;
    font-size: 9pt;
    color: #666;
  }
  @top-right {
    content: "FoundIt — WAICY 2026";
    font-family: Georgia, serif;
    font-size: 8pt;
    color: #999;
  }
}
body {
  font-family: Georgia, "Times New Roman", serif;
  font-size: 10.5pt;
  line-height: 1.55;
  color: #1a1a1a;
}
h1 {
  font-size: 22pt;
  line-height: 1.2;
  margin: 0 0 6px;
  letter-spacing: -0.01em;
}
h1 + p strong { font-size: 11pt; color: #444; }
h2 {
  font-size: 14pt;
  margin: 26px 0 8px;
  padding-bottom: 4px;
  border-bottom: 1.5px solid #d0d0d0;
  page-break-after: avoid;
}
h3 {
  font-size: 11.5pt;
  margin: 18px 0 6px;
  page-break-after: avoid;
}
p { margin: 0 0 9px; text-align: left; }
table {
  border-collapse: collapse;
  width: 100%;
  margin: 10px 0 14px;
  font-size: 9pt;
  page-break-inside: avoid;
}
th, td {
  border: 1px solid #cfcfcf;
  padding: 5px 7px;
  text-align: left;
  vertical-align: top;
}
th { background: #f2f4f7; font-weight: 700; }
tr:nth-child(even) td { background: #fafbfc; }
blockquote {
  margin: 10px 0;
  padding: 8px 14px;
  background: #f6f8fa;
  border-left: 3px solid #4a6fa5;
  font-style: italic;
}
code {
  font-family: "DejaVu Sans Mono", monospace;
  font-size: 8.8pt;
  background: #f2f2f2;
  padding: 1px 4px;
  border-radius: 3px;
}
pre {
  background: #f6f8fa;
  padding: 10px;
  border-radius: 4px;
  overflow-wrap: break-word;
  white-space: pre-wrap;
  font-size: 8.5pt;
}
hr { border: 0; border-top: 1px solid #ddd; margin: 22px 0; }
ul, ol { margin: 0 0 10px; padding-left: 22px; }
li { margin-bottom: 4px; }
strong { font-weight: 700; }
h2 { break-before: auto; }
"""


def convert() -> Path:
    text = SOURCE.read_text(encoding="utf-8")

    # The document is a draft: unresolved link placeholders are marked with
    # underscores. Highlight them so they cannot reach a judge unnoticed.
    text = re.sub(
        r"_\[(insert[^\]]*)\]_",
        r'<mark style="background:#ffe58a;padding:1px 3px">[TODO: \1]</mark>',
        text,
    )

    body = markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "sane_lists", "attr_list"],
    )
    html = f"""<!doctype html><html><head><meta charset="utf-8">
    <title>FoundIt — WAICY 2026</title>
    <style>{CSS}</style></head><body>{body}</body></html>"""

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    HTML(string=html, base_url=str(ROOT)).write_pdf(str(OUTPUT))
    return OUTPUT


if __name__ == "__main__":
    path = convert()
    size = path.stat().st_size
    print(f"wrote {path.relative_to(ROOT)} ({size / 1024:.0f} KB)")
    sys.exit(0)
