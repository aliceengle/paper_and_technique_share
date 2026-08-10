#!/usr/bin/env python3
"""Build a static HTML page for the speculative decoding technique report."""

from __future__ import annotations

import argparse
import html
import re
import unicodedata
from pathlib import Path

from bs4 import BeautifulSoup
from markdown_it import MarkdownIt


TECHNIQUE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    TECHNIQUE_DIR
    / "speculative_decoding_technique_comparison_glm52_report_20260721.md"
)
DEFAULT_OUTPUT_DIR = (
    TECHNIQUE_DIR
    / "html/speculative-decoding-technique-comparison-glm52-260721"
)


def protect_math(markdown: str) -> tuple[str, list[str]]:
    placeholders: list[str] = []

    def marker(index: int) -> str:
        return f"@@SPECDEC_MATH_PLACEHOLDER_{index}@@"

    def protect_segment(segment: str) -> str:
        def block_repl(match: re.Match[str]) -> str:
            placeholders.append(match.group(0))
            return f"\n\n{marker(len(placeholders) - 1)}\n\n"

        segment = re.sub(r"\$\$.*?\$\$", block_repl, segment, flags=re.S)

        def inline_repl(match: re.Match[str]) -> str:
            placeholders.append(match.group(0))
            return marker(len(placeholders) - 1)

        return re.sub(r"(?<!\\)\$(?!\$)(.+?)(?<!\\)\$", inline_repl, segment, flags=re.S)

    parts = re.split(r"(```[\s\S]*?```)", markdown)
    for i, part in enumerate(parts):
        if not part.startswith("```"):
            parts[i] = protect_segment(part)
    return "".join(parts), placeholders


def restore_math(rendered: str, placeholders: list[str]) -> str:
    for idx, value in enumerate(placeholders):
        rendered = rendered.replace(
            f"@@SPECDEC_MATH_PLACEHOLDER_{idx}@@",
            html.escape(value, quote=False),
        )
    return rendered


def slugify(text: str, seen: set[str]) -> str:
    normalized = unicodedata.normalize("NFKD", text).lower()
    slug = re.sub(r"[^\w\u4e00-\u9fff]+", "-", normalized, flags=re.U).strip("-")
    slug = slug or "section"
    base = slug
    n = 2
    while slug in seen:
        slug = f"{base}-{n}"
        n += 1
    seen.add(slug)
    return slug


def rewrite_soup(body_html: str) -> tuple[str, str, str]:
    soup = BeautifulSoup(body_html, "html.parser")
    title = "投机采样解码技术报告"
    first_h1 = soup.find("h1")
    if first_h1:
        title = first_h1.get_text(" ", strip=True)
        first_h1.decompose()

    seen: set[str] = set()
    toc_items: list[tuple[int, str, str]] = []
    for heading in soup.find_all(["h2", "h3"]):
        text = heading.get_text(" ", strip=True)
        if not text:
            continue
        hid = slugify(text, seen)
        heading["id"] = hid
        toc_items.append((int(heading.name[1]), hid, text))

    for code in soup.select("pre > code.language-mermaid"):
        pre = code.parent
        div = soup.new_tag("div")
        div["class"] = "mermaid"
        div.string = code.get_text()
        pre.replace_with(div)

    for table in soup.find_all("table"):
        wrapper = soup.new_tag("div")
        wrapper["class"] = "table-wrap"
        table.wrap(wrapper)

    toc_html = "\n".join(
        f'<li class="level-{level}"><a href="#{hid}">{html.escape(text)}</a></li>'
        for level, hid, text in toc_items
    )
    return title, str(soup), toc_html


def render_html(title: str, article: str, toc_html: str) -> str:
    description = "Speculative decoding 技术路线、算法演进、测试对比与 GLM-5.2 选型报告"
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{html.escape(title)}</title>
  <meta name="description" content="{html.escape(description)}">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Noto+Sans+SC:wght@300;400;500;600;700&family=Source+Serif+4:opsz,wght@8..60,500;8..60,600;8..60,700&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.21/dist/katex.min.css">
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.21/dist/katex.min.js"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.21/dist/contrib/auto-render.min.js"
    onload="renderMathInElement(document.body, {{delimiters:[{{left:'$$',right:'$$',display:true}},{{left:'$',right:'$',display:false}}], throwOnError:false}});"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>
  <style>
    :root {{
      --bg: #f5f7fa;
      --panel: #ffffff;
      --panel-soft: #f8fafc;
      --ink: #14202b;
      --ink-2: #435260;
      --ink-3: #7d8996;
      --border: #dbe4ec;
      --accent: #176f73;
      --accent-2: #476fa8;
      --accent-bg: rgba(23, 111, 115, .08);
      --measure: 1180px;
      --font-serif: "Source Serif 4", "Noto Sans SC", Georgia, serif;
      --font-sans: "Noto Sans SC", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      --font-mono: "IBM Plex Mono", "Noto Sans SC", monospace;
    }}
    *, *::before, *::after {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; font-size: 17px; }}
    body {{
      margin: 0;
      color: var(--ink-2);
      background: linear-gradient(180deg, #fbfdff 0, var(--bg) 520px), var(--bg);
      font-family: var(--font-sans);
      line-height: 1.76;
      -webkit-font-smoothing: antialiased;
    }}
    a {{ color: var(--accent); text-decoration-color: rgba(23,111,115,.28); text-underline-offset: 3px; }}
    a:hover {{ text-decoration-color: var(--accent); }}
    .progress-bar {{ position: fixed; inset: 0 0 auto 0; height: 3px; z-index: 100; }}
    .progress-bar .fill {{ height: 100%; width: 0; background: linear-gradient(90deg, var(--accent), var(--accent-2)); }}
    .page {{ max-width: var(--measure); margin: 0 auto; padding: 50px 28px 76px; }}
    .hero {{ padding: 24px 0 30px; border-bottom: 1px solid var(--border); margin-bottom: 24px; }}
    .eyebrow {{
      color: var(--accent);
      font: 600 .78rem/1.4 var(--font-mono);
      letter-spacing: .03em;
      text-transform: uppercase;
      margin-bottom: .6rem;
    }}
    h1, h2, h3, h4 {{ color: var(--ink); font-family: var(--font-serif); line-height: 1.34; }}
    h1 {{ max-width: 940px; margin: 0 0 .9rem; font-size: clamp(2rem, 4vw, 3.35rem); font-weight: 700; }}
    .lead {{ max-width: 850px; margin: 0; color: var(--ink-2); font-size: 1.04rem; }}
    .topline {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 18px; }}
    .pill {{
      display: inline-flex;
      align-items: center;
      min-height: 30px;
      padding: 4px 10px;
      border: 1px solid var(--border);
      border-radius: 999px;
      background: var(--panel);
      color: var(--ink-2);
      font-size: .86rem;
    }}
    .article-body {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 28px;
      box-shadow: 0 16px 40px rgba(20, 32, 43, .06);
    }}
    h2 {{ margin: 2.7rem 0 1rem; padding-bottom: .6rem; border-bottom: 1px solid var(--border); font-size: 1.46rem; }}
    .article-body > h2:first-child {{ margin-top: 0; }}
    h3 {{ margin: 1.8rem 0 .65rem; font-size: 1.08rem; }}
    h4 {{ margin: 1.25rem 0 .45rem; font-size: 1rem; }}
    p {{ margin: 0 0 1rem; }}
    blockquote {{
      margin: 1rem 0 1.2rem;
      padding: .85rem 1rem;
      border-left: 4px solid var(--accent);
      background: var(--accent-bg);
      border-radius: 0 8px 8px 0;
    }}
    code {{
      font-family: var(--font-mono);
      font-size: .86em;
      background: #edf4f7;
      color: #20303c;
      border: 1px solid rgba(190,205,218,.62);
      border-radius: 4px;
      padding: 1px 5px;
    }}
    pre {{
      overflow-x: auto;
      padding: 1rem;
      background: #101820;
      color: #e8eef3;
      border-radius: 8px;
      line-height: 1.58;
    }}
    pre code {{ background: transparent; border: 0; color: inherit; padding: 0; }}
    ul, ol {{ padding-left: 1.35rem; margin: .35rem 0 1.15rem; }}
    li {{ margin: .2rem 0; }}
    .table-wrap {{
      overflow-x: auto;
      margin: 1rem 0 1.35rem;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: #fff;
    }}
    table {{ width: 100%; min-width: 760px; border-collapse: collapse; font-size: .91rem; }}
    th, td {{ padding: .64rem .72rem; border-bottom: 1px solid var(--border); vertical-align: top; }}
    th {{ background: var(--panel-soft); color: var(--ink); font-weight: 600; text-align: left; }}
    tr:nth-child(even) td {{ background: #fbfcfd; }}
    tr:last-child td {{ border-bottom: 0; }}
    .mermaid {{
      margin: 1rem 0 1.35rem;
      padding: 1rem;
      background: var(--panel-soft);
      border: 1px solid var(--border);
      border-radius: 10px;
      overflow-x: auto;
      text-align: center;
    }}
    .mermaid svg {{
      display: block;
      max-width: 100%;
      height: auto;
      margin: 0 auto;
    }}
    .toc {{
      position: sticky;
      top: 18px;
      float: right;
      width: 250px;
      max-height: calc(100vh - 36px);
      overflow: auto;
      margin: 0 -282px 20px 24px;
      padding: 14px;
      border: 1px solid var(--border);
      border-radius: 10px;
      background: rgba(255, 255, 255, .9);
      backdrop-filter: blur(10px);
      font-size: .78rem;
    }}
    .toc-label {{ color: var(--ink-3); font: 600 .72rem/1 var(--font-mono); margin-bottom: .6rem; }}
    .toc ul {{ list-style: none; margin: 0; padding: 0; }}
    .toc a {{ display: block; padding: .22rem 0; color: var(--ink-3); text-decoration: none; }}
    .toc li.level-3 a {{ padding-left: .8rem; font-size: .74rem; }}
    .toc a.active {{ color: var(--accent); }}
    @media (max-width: 1720px) {{ .toc {{ display: none; }} }}
    @media (max-width: 720px) {{
      html {{ font-size: 16px; }}
      .page {{ padding: 32px 14px 56px; }}
      .article-body {{ padding: 18px; border-radius: 10px; }}
      table {{ min-width: 640px; }}
    }}
    @media print {{
      .progress-bar, .toc {{ display: none !important; }}
      body {{ background: #fff; }}
      .page {{ max-width: none; padding: 0; }}
      .article-body {{ box-shadow: none; border: 0; padding: 0; }}
    }}
  </style>
</head>
<body>
  <div class="progress-bar"><div class="fill" id="progressFill"></div></div>
  <main class="page">
    <header class="hero">
      <div class="eyebrow">Speculative Decoding / GLM-5.2</div>
      <h1>{html.escape(title)}</h1>
      <p class="lead">{html.escape(description)}</p>
      <div class="topline">
        <span class="pill">Medusa / EAGLE</span>
        <span class="pill">native MTP / FastMTP</span>
        <span class="pill">DFlash / DSpark</span>
        <span class="pill">GLM-5.2 选型</span>
      </div>
    </header>
    <nav class="toc" aria-label="文章目录">
      <div class="toc-label">目录</div>
      <ul id="tocList">
        {toc_html}
      </ul>
    </nav>
    <article class="article-body">
      {article}
    </article>
  </main>
  <script>
    window.addEventListener('DOMContentLoaded', () => {{
      if (window.mermaid) {{
        mermaid.initialize({{ startOnLoad: true, theme: 'neutral', securityLevel: 'strict' }});
      }}
    }});
    const fill = document.getElementById('progressFill');
    window.addEventListener('scroll', () => {{
      const h = document.documentElement.scrollHeight - window.innerHeight;
      fill.style.width = h > 0 ? `${{window.scrollY / h * 100}}%` : '0%';
    }}, {{ passive: true }});
    const links = document.querySelectorAll('#tocList a');
    const sections = [];
    links.forEach(a => {{
      const el = document.getElementById(a.getAttribute('href').slice(1));
      if (el) sections.push({{ el, a }});
    }});
    const observer = new IntersectionObserver((entries) => {{
      entries.forEach(entry => {{
        if (entry.isIntersecting) {{
          links.forEach(a => a.classList.remove('active'));
          const hit = sections.find(s => s.el === entry.target);
          if (hit) hit.a.classList.add('active');
        }}
      }});
    }}, {{ threshold: 0.08, rootMargin: '-72px 0px -64% 0px' }});
    sections.forEach(s => observer.observe(s.el));
  </script>
</body>
</html>
"""


def build(source: Path, output_dir: Path) -> Path:
    source_text = source.read_text(encoding="utf-8")
    protected, placeholders = protect_math(source_text)
    rendered = MarkdownIt(
        "default",
        {"html": False, "linkify": False, "typographer": False},
    ).render(protected)
    rendered = restore_math(rendered, placeholders)
    title, article, toc_html = rewrite_soup(rendered)

    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / "index.html"
    html_path.write_text(render_html(title, article, toc_html), encoding="utf-8")
    return html_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    html_path = build(args.source, args.output_dir)
    print(html_path)


if __name__ == "__main__":
    main()
