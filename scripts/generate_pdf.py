"""
Compiles docs/research_note.md into an academic publication-grade PDF report
using Microsoft Edge headless renderer.
"""

import os
import re
import subprocess
from pathlib import Path
import markdown_it

ROOT_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT_DIR / "docs"
EDGE_EXE = Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe")


def markdown_to_html(md_text: str) -> str:
    """Convert academic markdown report into publication-ready HTML with CSS styling."""
    abs_plot_path = (ROOT_DIR / "results" / "plots" / "comparison_metrics.png").resolve().as_uri()

    # Pre-process Mermaid blocks into clean styled HTML diagrams
    mermaid_pattern = re.compile(r"```mermaid\s+flowchart LR\s+(.*?)\s+```", re.DOTALL)

    def replace_mermaid(match):
        return """
<div class="workflow-box">
  <div class="workflow-step"><strong>1. Residual Probe</strong><br>Dev Set (10 SV / 10 EN)</div>
  <div class="workflow-arrow">&rarr;</div>
  <div class="workflow-step"><strong>2. Direction Vector</strong><br>Layers 14 & 16 ($d_l$)</div>
  <div class="workflow-arrow">&rarr;</div>
  <div class="workflow-step"><strong>3. Interventions</strong><br>WSP vs. Random Matched</div>
  <div class="workflow-arrow">&rarr;</div>
  <div class="workflow-step"><strong>4. Evaluation</strong><br>30 SV / 20 EN Held-Out</div>
</div>
"""

    processed_md = mermaid_pattern.sub(replace_mermaid, md_text)

    # Initialize markdown parser with GFM tables
    md = markdown_it.MarkdownIt("gfm-like").disable("linkify")
    rendered_body = md.render(processed_md)

    # Replace relative plot image path with file URI
    rendered_body = re.sub(
        r'src="(?:\.\./results/plots/comparison_metrics\.png|results/plots/comparison_metrics\.png)"',
        f'src="{abs_plot_path}"',
        rendered_body,
    )

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Weight-Space Programming for SE.LLMA - Research Note</title>
<style>
  @page {{
    size: A4;
    margin: 14mm 14mm 14mm 14mm;
    @bottom-center {{
      content: counter(page);
      font-size: 8pt;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
      color: #64748b;
    }}
  }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    color: #1e293b;
    line-height: 1.42;
    font-size: 9pt;
    margin: 0;
    padding: 0;
  }}
  h1 {{
    font-size: 15pt;
    font-weight: 800;
    text-align: center;
    margin-bottom: 4px;
    color: #0f172a;
    line-height: 1.2;
  }}
  h2 {{
    font-size: 10.5pt;
    font-weight: 700;
    color: #0f172a;
    border-bottom: 1px solid #cbd5e1;
    padding-bottom: 2px;
    margin-top: 10px;
    margin-bottom: 4px;
    text-transform: uppercase;
    letter-spacing: 0.3px;
    page-break-after: avoid;
  }}
  h3 {{
    font-size: 9.5pt;
    font-weight: 600;
    color: #1e293b;
    margin-top: 7px;
    margin-bottom: 2px;
    page-break-after: avoid;
  }}
  h4 {{
    font-size: 9pt;
    font-weight: 600;
    color: #334155;
    margin-top: 6px;
    margin-bottom: 2px;
    page-break-after: avoid;
  }}
  p {{
    margin-top: 0;
    margin-bottom: 5px;
    text-align: justify;
  }}
  blockquote {{
    background: #f1f5f9;
    border-left: 3px solid #0284c7;
    margin: 6px 0;
    padding: 4px 10px;
    font-style: italic;
    color: #1e293b;
    border-radius: 0 4px 4px 0;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    margin: 8px 0;
    font-size: 7.8pt;
    page-break-inside: avoid;
  }}
  th, td {{
    border: 1px solid #cbd5e1;
    padding: 4px 6px;
    text-align: left;
  }}
  th {{
    background-color: #f8fafc;
    font-weight: 700;
    color: #0f172a;
  }}
  tr:nth-child(even) {{
    background-color: #f8fafc;
  }}
  code {{
    background-color: #f1f5f9;
    padding: 1px 3px;
    border-radius: 3px;
    font-family: Consolas, "Courier New", monospace;
    font-size: 8pt;
    color: #0369a1;
  }}
  img {{
    max-width: 82%;
    height: auto;
    display: block;
    margin: 6px auto 2px auto;
    border: 1px solid #e2e8f0;
    border-radius: 4px;
    page-break-inside: avoid;
  }}
  .figure-caption {{
    text-align: center;
    font-size: 7.5pt;
    color: #64748b;
    margin-top: 2px;
    margin-bottom: 8px;
    font-style: italic;
  }}
  ul, ol {{
    margin-top: 0;
    margin-bottom: 5px;
    padding-left: 16px;
  }}
  li {{
    margin-bottom: 2px;
  }}
  hr {{
    border: 0;
    border-top: 1px solid #e2e8f0;
    margin: 8px 0;
  }}
  .workflow-box {{
    display: flex;
    align-items: center;
    justify-content: space-around;
    background: #f8fafc;
    border: 1px solid #cbd5e1;
    border-radius: 5px;
    padding: 6px 10px;
    margin: 6px 0;
    page-break-inside: avoid;
  }}
  .workflow-step {{
    background: #ffffff;
    border: 1px solid #94a3b8;
    border-radius: 4px;
    padding: 4px 6px;
    text-align: center;
    font-size: 7.5pt;
    line-height: 1.2;
  }}
  .workflow-arrow {{
    font-size: 12pt;
    color: #0284c7;
    font-weight: bold;
  }}
</style>
</head>
<body>
{rendered_body}
</body>
</html>
"""
    return full_html


def build_pdf():
    md_file = DOCS_DIR / "research_note.md"
    html_file = DOCS_DIR / "report.html"
    pdf_file = DOCS_DIR / "report.pdf"

    if not md_file.exists():
        print(f"Error: {md_file} does not exist.")
        return

    print("Generating publication HTML from Markdown...")
    with open(md_file, "r", encoding="utf-8") as f:
        md_text = f.read()

    html_text = markdown_to_html(md_text)
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(html_text)

    print(f"HTML generated at: {html_file}")

    if not EDGE_EXE.exists():
        print(f"Edge executable not found at: {EDGE_EXE}")
        return

    print("Compiling PDF via Microsoft Edge headless...")
    cmd = [
        str(EDGE_EXE),
        "--headless",
        "--disable-gpu",
        "--run-all-compositor-stages-before-draw",
        f"--print-to-pdf={pdf_file.resolve()}",
        str(html_file.resolve()),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if pdf_file.exists():
        size_mb = pdf_file.stat().st_size / (1024 * 1024)
        print(f"PDF successfully compiled: {pdf_file} ({size_mb:.2f} MB)")
    else:
        print("PDF compilation failed:", res.stderr)


if __name__ == "__main__":
    build_pdf()
