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


def get_display_equation_html(raw: str) -> str:
    """Return publication-grade semantic HTML for display equations."""
    eq1_html = """
<div class="math-display">
  <i>d</i><sub><i>l</i></sub> = 
  <span class="math-fraction">
    <span class="numerator"><i>h&#772;</i><sub>sv</sub> &minus; <i>h&#772;</i><sub>en</sub></span>
    <span class="denominator">&Vert;<i>h&#772;</i><sub>sv</sub> &minus; <i>h&#772;</i><sub>en</sub>&Vert;<sub>2</sub></span>
  </span>
  &isin; &#8477;<sup><i>d</i></sup>
</div>"""

    eq2_html = """
<div class="math-display">
  <i>u</i><sub><i>l</i>, <i>j</i></sub> = 
  <span class="math-cases">
    <span class="cases-brace">{</span>
    <span class="cases-content">
      <span class="cases-row"><span><i>a</i><sub><i>l</i>, <i>j</i></sub></span><span>if <i>j</i> &isin; <i>T</i><sub><i>l</i></sub></span></span>
      <span class="cases-row"><span>0</span><span>otherwise</span></span>
    </span>
  </span>,
  &emsp;&emsp;
  <i>u</i><sub><i>l</i></sub> &larr; 
  <span class="math-fraction">
    <span class="numerator"><i>u</i><sub><i>l</i></sub></span>
    <span class="denominator">&Vert;<i>u</i><sub><i>l</i></sub>&Vert;<sub>2</sub></span>
  </span>
</div>"""

    eq3_html = """
<div class="math-display">
  &Delta;<i>W</i><sub>down</sub><sup>(<i>l</i>)</sup> = &alpha; &middot; <i>d</i><sub><i>l</i></sub> &otimes; <i>u</i><sub><i>l</i></sub><sup><i>T</i></sup> = &alpha; &middot; <i>d</i><sub><i>l</i></sub> <i>u</i><sub><i>l</i></sub><sup><i>T</i></sup> &isin; &#8477;<sup><i>d</i> &times; <i>d</i><sub>ff</sub></sup>
</div>"""

    if "d_l" in raw and ("bar{h}" in raw or "h_{sv}" in raw):
        return eq1_html
    elif "u_{l" in raw or "begin{cases}" in raw:
        return eq2_html
    elif "Delta W_{down}" in raw:
        return eq3_html
    return f'<div class="math-display">{raw}</div>'


def render_inline_math(html: str) -> str:
    """Convert inline LaTeX math expressions ($...$) to clean semantic HTML."""
    exact_inline_map = [
        (
            r"$\mathcal{P} = \exp\left(-\frac{1}{N_{tok}}\sum \log P(x_i)\right)$",
            '<span class="math-inline"><i>P</i> = exp(&minus; <sup>1</sup>/<sub><i>N</i><sub>tok</sub></sub> &sum; log <i>P</i>(<i>x</i><sub><i>i</i></sub>))</span>',
        ),
        (
            r"$\|\Delta W_{down}^{(l)}\|_F = \alpha \|d_l\|_2 \|u_l\|_2 = 0.600$",
            '<span class="math-inline">&Vert;&Delta;<i>W</i><sub>down</sub><sup>(<i>l</i>)</sup>&Vert;<sub>F</sub> = &alpha;&Vert;<i>d</i><sub><i>l</i></sub>&Vert;<sub>2</sub>&Vert;<i>u</i><sub><i>l</i></sub>&Vert;<sub>2</sub> = 0.600</span>',
        ),
        (
            r"$\Delta W_{down}^{(l)}$",
            '<span class="math-inline">&Delta;<i>W</i><sub>down</sub><sup>(<i>l</i>)</sup></span>',
        ),
        (
            r"$W_{down} \in \mathbb{R}^{d \times d_{ff}}$",
            '<span class="math-inline"><i>W</i><sub>down</sub> &isin; &#8477;<sup><i>d</i> &times; <i>d</i><sub>ff</sub></sup></span>',
        ),
        (
            r"$W_{down}^{(l)}$",
            '<span class="math-inline"><i>W</i><sub>down</sub><sup>(<i>l</i>)</sup></span>',
        ),
        (
            r"$a_{l, j} = d_l^T W_{down}^{(l)}[:, j]$",
            '<span class="math-inline"><i>a</i><sub><i>l</i>, <i>j</i></sub> = <i>d</i><sub><i>l</i></sub><sup><i>T</i></sup> <i>W</i><sub>down</sub><sup>(<i>l</i>)</sup>[:, <i>j</i>]</span>',
        ),
        (
            r"$j \in \{1, \dots, d_{ff}\}$",
            '<span class="math-inline"><i>j</i> &isin; {1, &hellip;, <i>d</i><sub>ff</sub>}</span>',
        ),
        (
            r"$d_{rand} \sim \mathcal{N}(0, I_d)$",
            '<span class="math-inline"><i>d</i><sub>rand</sub> &sim; <i>N</i>(0, <i>I</i><sub><i>d</i></sub>)</span>',
        ),
        (
            r"$u_{rand} \in \mathbb{R}^{d_{ff}}$",
            '<span class="math-inline"><i>u</i><sub>rand</sub> &isin; &#8477;<sup><i>d</i><sub>ff</sub></sup></span>',
        ),
        (
            r"$h_l \leftarrow h_l + \beta d_l$",
            '<span class="math-inline"><i>h</i><sub><i>l</i></sub> &larr; <i>h</i><sub><i>l</i></sub> + &beta;<i>d</i><sub><i>l</i></sub></span>',
        ),
        (
            r"$\beta \in [0.5, 1.0, 2.0]$",
            '<span class="math-inline">&beta; &isin; [0.5, 1.0, 2.0]</span>',
        ),
        (
            r"$W_q, W_k, W_v$",
            '<span class="math-inline"><i>W</i><sub>q</sub>, <i>W</i><sub>k</sub>, <i>W</i><sub>v</sub></span>',
        ),
        (
            r"$u_l \in \mathbb{R}^{d_{ff}}$",
            '<span class="math-inline"><i>u</i><sub><i>l</i></sub> &isin; &#8477;<sup><i>d</i><sub>ff</sub></sup></span>',
        ),
        (
            r"$j \in \mathcal{T}_l$",
            '<span class="math-inline"><i>j</i> &isin; <i>T</i><sub><i>l</i></sub></span>',
        ),
        (r"$\mathcal{T}_l$", '<span class="math-inline"><i>T</i><sub><i>l</i></sub></span>'),
        (r"$W_{down}$", '<span class="math-inline"><i>W</i><sub>down</sub></span>'),
        (r"$W_{head}$", '<span class="math-inline"><i>W</i><sub>head</sub></span>'),
        (r"$W_o$", '<span class="math-inline"><i>W</i><sub>o</sub></span>'),
        (r"$\alpha = 0.6$", '<span class="math-inline">&alpha; = 0.6</span>'),
        (r"$\beta = 2.0$", '<span class="math-inline">&beta; = 2.0</span>'),
        (r"$\beta=2.0$", '<span class="math-inline">&beta; = 2.0</span>'),
        (r"$\approx 0.600$", '<span class="math-inline">&asymp; 0.600</span>'),
        (r"$\pm$", '<span class="math-inline">&plusmn;</span>'),
        (r"$\neq$", '<span class="math-inline">&ne;</span>'),
        (r"$d=896$", '<span class="math-inline"><i>d</i> = 896</span>'),
        (r"$d_{ff}=4864$", '<span class="math-inline"><i>d</i><sub>ff</sub> = 4864</span>'),
        (r"$d_l$", '<span class="math-inline"><i>d</i><sub><i>l</i></sub></span>'),
        (r"$u_l$", '<span class="math-inline"><i>u</i><sub><i>l</i></sub></span>'),
        (r"$k=48$", '<span class="math-inline"><i>k</i> = 48</span>'),
        (r"$k$", '<span class="math-inline"><i>k</i></span>'),
        (r"$|a_{l, j}|$", '<span class="math-inline">|<i>a</i><sub><i>l</i>, <i>j</i></sub>|</span>'),
        (r"$= 0.0$", '<span class="math-inline">= 0.0</span>'),
        (r"$0.600$", '<span class="math-inline">0.600</span>'),
    ]

    # Sort by length descending so longer patterns match first
    exact_inline_map.sort(key=lambda x: len(x[0]), reverse=True)

    for raw, repl in exact_inline_map:
        html = html.replace(raw, repl)

    # Dynamic regex replacements for $N=...$
    html = re.sub(
        r"\$N\s*=\s*(\d+)\$",
        r'<span class="math-inline"><i>N</i> = \1</span>',
        html,
    )

    # Fallback generic inline math cleanup
    def clean_generic_math(match):
        inner = match.group(1).strip()
        inner = inner.replace(r"\alpha", "&alpha;")
        inner = inner.replace(r"\beta", "&beta;")
        inner = inner.replace(r"\Delta", "&Delta;")
        inner = inner.replace(r"\sim", "&sim;")
        inner = inner.replace(r"\approx", "&asymp;")
        inner = inner.replace(r"\pm", "&plusmn;")
        inner = inner.replace(r"\neq", "&ne;")
        inner = inner.replace(r"\in", "&isin;")
        inner = inner.replace(r"\mathbb{R}", "&#8477;")
        inner = re.sub(r"([A-Za-z]+)_\{([^}]+)\}", r"<i>\1</i><sub>\2</sub>", inner)
        inner = re.sub(r"([A-Za-z]+)_([A-Za-z0-9]+)", r"<i>\1</i><sub>\2</sub>", inner)
        return f'<span class="math-inline">{inner}</span>'

    html = re.sub(r"\$([^\$\n]+?)\$", clean_generic_math, html)
    return html


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
  <div class="workflow-step"><strong>2. Direction Vector</strong><br>Layers 14 & 16 (<i>d</i><sub><i>l</i></sub>)</div>
  <div class="workflow-arrow">&rarr;</div>
  <div class="workflow-step"><strong>3. Interventions</strong><br>WSP vs. Random Matched</div>
  <div class="workflow-arrow">&rarr;</div>
  <div class="workflow-step"><strong>4. Evaluation</strong><br>30 SV / 20 EN Held-Out</div>
</div>
"""

    processed_md = mermaid_pattern.sub(replace_mermaid, md_text)

    # 1. Protect all display math blocks ($$...$$) using unique placeholders
    # so they NEVER break Markdown list/block parsing
    display_placeholders = {}

    def extract_display_math(match):
        idx = len(display_placeholders)
        placeholder_key = f"DISPLAYMATHPLACEHOLDER{idx}XYZ"
        raw_eq = match.group(1).strip()
        display_placeholders[placeholder_key] = get_display_equation_html(raw_eq)
        return f"\n\n{placeholder_key}\n\n"

    processed_md = re.sub(r"\$\$(.*?)\$\$", extract_display_math, processed_md, flags=re.DOTALL)

    # 2. Render all inline math expressions to clean inline HTML BEFORE Markdown parsing
    # so underscores inside math (e.g. d_{rand}) are never mangled into <em> italics
    processed_md = render_inline_math(processed_md)

    # 3. Render Markdown to HTML via markdown_it (GFM-like)
    md = markdown_it.MarkdownIt("gfm-like").disable("linkify")
    rendered_body = md.render(processed_md)

    # 4. Substitute display math placeholders back into the HTML
    for placeholder_key, eq_html in display_placeholders.items():
        rendered_body = rendered_body.replace(f"<p>{placeholder_key}</p>", eq_html)
        rendered_body = rendered_body.replace(placeholder_key, eq_html)

    # 5. Replace relative plot image path with file URI
    rendered_body = re.sub(
        r'src="(?:\.\./results/plots/comparison_metrics\.png|results/plots/comparison_metrics\.png)"',
        f'src="{abs_plot_path}"',
        rendered_body,
    )

    # 6. Ensure link to AI_USAGE.md points to online repository for standalone PDF viewing
    rendered_body = re.sub(
        r'href="(?:\.\./AI_USAGE\.md|AI_USAGE\.md)"',
        'href="https://github.com/felixsuarez0727/weight-space-programming-sellma/blob/main/AI_USAGE.md"',
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
    margin: 12mm 12mm 12mm 12mm;
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
    line-height: 1.40;
    font-size: 8.8pt;
    margin: 0;
    padding: 0;
  }}
  h1 {{
    font-size: 14.5pt;
    font-weight: 800;
    text-align: center;
    margin-bottom: 3px;
    color: #0f172a;
    line-height: 1.2;
  }}
  h2 {{
    font-size: 10.2pt;
    font-weight: 700;
    color: #0f172a;
    border-bottom: 1px solid #cbd5e1;
    padding-bottom: 2px;
    margin-top: 9px;
    margin-bottom: 3px;
    text-transform: uppercase;
    letter-spacing: 0.3px;
    page-break-after: avoid;
  }}
  h3 {{
    font-size: 9.2pt;
    font-weight: 600;
    color: #1e293b;
    margin-top: 6px;
    margin-bottom: 2px;
    page-break-after: avoid;
  }}
  h4 {{
    font-size: 8.8pt;
    font-weight: 600;
    color: #334155;
    margin-top: 5px;
    margin-bottom: 2px;
    page-break-after: avoid;
  }}
  p {{
    margin-top: 0;
    margin-bottom: 4px;
    text-align: justify;
  }}
  blockquote {{
    background: #f1f5f9;
    border-left: 3px solid #0284c7;
    margin: 5px 0;
    padding: 4px 8px;
    font-style: italic;
    color: #1e293b;
    border-radius: 0 4px 4px 0;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    margin: 6px 0;
    font-size: 7.2pt;
    page-break-inside: avoid;
  }}
  th, td {{
    border: 1px solid #cbd5e1;
    padding: 3px 4px;
    text-align: left;
  }}
  th {{
    background-color: #f1f5f9;
    font-weight: 700;
    color: #0f172a;
    font-size: 7.2pt;
  }}
  tr:nth-child(even) {{
    background-color: #f8fafc;
  }}
  code {{
    background-color: #f1f5f9;
    padding: 1px 3px;
    border-radius: 3px;
    font-family: Consolas, "Courier New", monospace;
    font-size: 7.8pt;
    color: #0369a1;
  }}
  pre {{
    background-color: #f8fafc;
    border: 1px solid #e2e8f0;
    padding: 6px 8px;
    border-radius: 4px;
    font-size: 7.6pt;
    line-height: 1.25;
    overflow-x: auto;
    margin: 4px 0;
  }}
  img {{
    max-width: 100%;
    height: auto;
    display: block;
    margin: 6px auto;
    border-radius: 4px;
    border: 1px solid #e2e8f0;
  }}
  .figure-caption {{
    text-align: center;
    font-size: 7.5pt;
    color: #64748b;
    margin-top: 2px;
    margin-bottom: 6px;
    font-style: italic;
  }}
  .workflow-box {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 4px;
    padding: 6px 10px;
    margin: 6px 0;
    font-size: 7.5pt;
  }}
  .workflow-step {{
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 3px;
    padding: 3px 6px;
    text-align: center;
    flex: 1;
    margin: 0 3px;
  }}
  .workflow-arrow {{
    font-size: 11pt;
    color: #0284c7;
    font-weight: bold;
    padding: 0 2px;
  }}
  .math-display {{
    display: flex;
    justify-content: center;
    align-items: center;
    margin: 6px 0;
    font-size: 8.8pt;
    font-family: "Cambria Math", "Latin Modern Math", "STIX Two Math", "Times New Roman", serif;
  }}
  .math-fraction {{
    display: inline-flex;
    flex-direction: column;
    vertical-align: middle;
    text-align: center;
    padding: 0 4px;
    font-size: 0.9em;
  }}
  .math-fraction .numerator {{
    border-bottom: 1px solid #1e293b;
    padding-bottom: 1px;
  }}
  .math-fraction .denominator {{
    padding-top: 1px;
  }}
  .math-cases {{
    display: inline-flex;
    align-items: center;
    margin: 0 4px;
  }}
  .cases-brace {{
    font-size: 2.1em;
    font-weight: 200;
    line-height: 1;
    margin-right: 3px;
  }}
  .cases-content {{
    display: inline-flex;
    flex-direction: column;
    text-align: left;
    font-size: 0.9em;
    gap: 2px;
  }}
  .cases-row {{
    display: flex;
    gap: 14px;
  }}
  .math-inline {{
    font-family: "Cambria Math", "Latin Modern Math", "STIX Two Math", "Times New Roman", serif;
    font-size: 8.8pt;
  }}
  hr {{
    border: none;
    border-top: 1px solid #e2e8f0;
    margin: 7px 0;
  }}
  a {{
    color: #0284c7;
    text-decoration: none;
  }}
  a:hover {{
    text-decoration: underline;
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
    html_file = DOCS_DIR / "Weight_Space_Programming_SELLMA_Research_Report.html"
    pdf_file = DOCS_DIR / "Weight_Space_Programming_SELLMA_Research_Report.pdf"

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
    if pdf_file.exists():
        try:
            pdf_file.unlink()
        except Exception:
            pass

    cmd = [
        str(EDGE_EXE),
        "--headless",
        "--disable-gpu",
        "--run-all-compositor-stages-before-draw",
        "--no-pdf-header-footer",
        "--print-to-pdf-no-header",
        f"--print-to-pdf={pdf_file.resolve()}",
        str(html_file.resolve().as_uri()),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if pdf_file.exists():
        size_mb = pdf_file.stat().st_size / (1024 * 1024)
        print(f"PDF successfully compiled: {pdf_file} ({size_mb:.2f} MB)")
    else:
        print("PDF compilation failed:", res.stderr)


if __name__ == "__main__":
    build_pdf()
