#!/usr/bin/env python3
"""Build a shareable static HTML page for the Kimi technology timeline."""

from __future__ import annotations

import argparse
import html
import re
import shutil
import unicodedata
from pathlib import Path

from bs4 import BeautifulSoup
from markdown_it import MarkdownIt


ROOT = Path(__file__).resolve().parents[4]
TECHNIQUE_DIR = ROOT / "contexts/kimi_k2_7/Technique"
DEFAULT_SOURCE = TECHNIQUE_DIR / "kimi_series_technology_timeline_20260724.md"
DEFAULT_OUTPUT_DIR = TECHNIQUE_DIR / "html/kimi-series-technology-timeline-2607"
DEFAULT_ASSET_DIR = TECHNIQUE_DIR / "assets/kimi_series"


def protect_math(markdown: str) -> tuple[str, list[str]]:
    """Protect LaTeX spans so Markdown emphasis parsing does not corrupt underscores."""
    placeholders: list[str] = []

    def protect_segment(segment: str) -> str:
        def block_repl(match: re.Match[str]) -> str:
            placeholders.append(match.group(0))
            return f"\n\nKIMI_MATH_PLACEHOLDER_{len(placeholders) - 1}\n\n"

        segment = re.sub(r"\$\$.*?\$\$", block_repl, segment, flags=re.S)

        def inline_repl(match: re.Match[str]) -> str:
            placeholders.append(match.group(0))
            return f"KIMI_MATH_PLACEHOLDER_{len(placeholders) - 1}"

        return re.sub(r"(?<!\\)\$(?!\$)(.+?)(?<!\\)\$", inline_repl, segment, flags=re.S)

    parts = re.split(r"(```[\s\S]*?```)", markdown)
    for i, part in enumerate(parts):
        if not part.startswith("```"):
            parts[i] = protect_segment(part)
    return "".join(parts), placeholders


def restore_math(rendered: str, placeholders: list[str]) -> str:
    for idx, value in enumerate(placeholders):
        rendered = rendered.replace(f"KIMI_MATH_PLACEHOLDER_{idx}", html.escape(value, quote=False))
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


def rewrite_soup(body_html: str) -> tuple[str, str, str, str]:
    soup = BeautifulSoup(body_html, "html.parser")

    title = "Kimi 系列技术演进研究"
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

    for img in soup.find_all("img"):
        img["loading"] = "lazy"
        img["decoding"] = "async"

    toc_html = "\n".join(
        f'<li class="level-{level}"><a href="#{hid}">{html.escape(text)}</a></li>'
        for level, hid, text in toc_items
    )
    description = "Kimi 系列技术演进研究：时间线、推理效率、RL 后训练与 Agent 编排"
    return title, description, str(soup), toc_html


def render_html(title: str, description: str, article: str, toc_html: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{html.escape(title)}</title>
  <meta name="description" content="{html.escape(description)}">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Noto+Sans+SC:wght@300;400;500;600;700&family=Source+Serif+4:opsz,wght@8..60,400;8..60,500;8..60,600;8..60,700&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.21/dist/katex.min.css">
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.21/dist/katex.min.js"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.21/dist/contrib/auto-render.min.js"
    onload="renderMathInElement(document.body, {{delimiters:[{{left:'$$',right:'$$',display:true}},{{left:'$',right:'$',display:false}}], throwOnError:false}});"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>
  <style>
    :root {{
      --bg: #ffffff;
      --bg-soft: #f7f9fb;
      --bg-code: #f1f5f8;
      --ink: #17202a;
      --ink-2: #485463;
      --ink-3: #7d8794;
      --border: #dfe6ec;
      --accent: #276a73;
      --accent-2: #496fa8;
      --accent-bg: rgba(39, 106, 115, 0.08);
      --measure: 1120px;
      --font-serif: "Source Serif 4", "Noto Sans SC", Georgia, serif;
      --font-sans: "Noto Sans SC", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      --font-mono: "IBM Plex Mono", "Noto Sans SC", monospace;
    }}
    *, *::before, *::after {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; font-size: 17px; }}
    body {{
      margin: 0;
      color: var(--ink-2);
      background: var(--bg);
      font-family: var(--font-sans);
      line-height: 1.78;
      -webkit-font-smoothing: antialiased;
    }}
    a {{ color: var(--accent); text-decoration-color: rgba(39,106,115,.28); text-underline-offset: 3px; }}
    a:hover {{ text-decoration-color: var(--accent); }}
    .progress-bar {{ position: fixed; inset: 0 0 auto 0; height: 3px; z-index: 50; background: transparent; }}
    .progress-bar .fill {{ height: 100%; width: 0; background: linear-gradient(90deg, var(--accent), var(--accent-2)); }}
    .shell {{ max-width: var(--measure); margin: 0 auto; padding: 56px 28px 80px; }}
    .article-header {{ margin-bottom: 2.4rem; padding-bottom: 1.6rem; border-bottom: 1px solid var(--border); }}
    .kicker {{ color: var(--accent); font: 600 .78rem/1.4 var(--font-mono); letter-spacing: .03em; text-transform: uppercase; }}
    h1, h2, h3, h4 {{ color: var(--ink); font-family: var(--font-serif); line-height: 1.34; }}
    h1 {{ margin: .5rem 0 .8rem; font-size: clamp(1.9rem, 3.6vw, 3rem); font-weight: 700; }}
    h2 {{ margin: 3rem 0 1rem; padding-bottom: .55rem; border-bottom: 1px solid var(--border); font-size: 1.52rem; }}
    h3 {{ margin: 2rem 0 .7rem; font-size: 1.15rem; }}
    h4 {{ margin: 1.4rem 0 .45rem; font-size: 1rem; }}
    p {{ margin: 0 0 1.05rem; }}
    blockquote {{
      margin: 1.1rem 0 1.35rem;
      padding: .85rem 1rem;
      border-left: 4px solid var(--accent);
      background: var(--accent-bg);
    }}
    code {{
      font-family: var(--font-mono);
      font-size: .86em;
      background: var(--bg-code);
      color: #20303c;
      border: 1px solid rgba(184,199,213,.55);
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
    .table-wrap {{ overflow-x: auto; margin: 1rem 0 1.45rem; border: 1px solid var(--border); border-radius: 8px; }}
    table {{ width: 100%; border-collapse: collapse; min-width: 720px; font-size: .92rem; }}
    th, td {{ padding: .68rem .78rem; border-bottom: 1px solid var(--border); vertical-align: top; }}
    th {{ background: var(--bg-soft); color: var(--ink); font-weight: 600; text-align: left; }}
    tr:last-child td {{ border-bottom: 0; }}
    img {{ max-width: 100%; display: block; margin: 1.2rem auto; border: 1px solid var(--border); border-radius: 8px; background: #fff; }}
    .mermaid {{
      margin: 1.2rem 0 1.6rem;
      padding: 1rem;
      background: var(--bg-soft);
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow-x: auto;
    }}
    .toc-rail {{
      position: fixed;
      left: max(16px, calc((100vw - var(--measure)) / 2 - 250px));
      top: 72px;
      width: 220px;
      max-height: calc(100vh - 96px);
      overflow: auto;
      padding-right: 8px;
      font-size: .78rem;
    }}
    .toc-label {{ color: var(--ink-3); font: 600 .72rem/1 var(--font-mono); margin-bottom: .7rem; }}
    .toc-rail ul {{ list-style: none; margin: 0; padding: 0; }}
    .toc-rail li {{ margin: 0; }}
    .toc-rail a {{
      display: block;
      padding: .26rem 0 .26rem .55rem;
      color: var(--ink-3);
      text-decoration: none;
      border-left: 2px solid transparent;
    }}
    .toc-rail li.level-3 a {{ padding-left: 1.1rem; font-size: .74rem; }}
    .toc-rail a.active {{ color: var(--accent); border-left-color: var(--accent); }}
    .byline {{ margin: 0; color: var(--ink-3); font-size: .92rem; }}
    .meta-line {{ color: var(--ink-3); font-size: .86rem; }}
    @media (max-width: 1500px) {{ .toc-rail {{ display: none; }} }}
    @media (max-width: 720px) {{
      html {{ font-size: 16px; }}
      .shell {{ padding: 42px 18px 64px; }}
      table {{ min-width: 620px; }}
    }}
    @media print {{
      .progress-bar, .toc-rail {{ display: none !important; }}
      .shell {{ max-width: none; padding: 0; }}
      a {{ color: inherit; }}
    }}
  </style>
</head>
<body>
  <div class="progress-bar"><div class="fill" id="progressFill"></div></div>
  <nav class="toc-rail" aria-label="文章目录">
    <div class="toc-label">目录</div>
    <ul id="tocList">
      {toc_html}
    </ul>
  </nav>
  <main class="shell">
    <header class="article-header">
      <div class="kicker">Kimi Series / Technology Timeline</div>
      <h1>{html.escape(title)}</h1>
      <p class="byline">2026-07-29 · 推理效率 · RL 后训练 · Agent 编排 · K3 / K4</p>
      <p class="meta-line">由 Markdown 生成的静态分享页；公式使用 KaTeX，流程图使用 Mermaid，截图资产随页面目录发布。</p>
    </header>
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
    const tocLinks = document.querySelectorAll('#tocList a');
    const sections = [];
    tocLinks.forEach(a => {{
      const el = document.getElementById(a.getAttribute('href').slice(1));
      if (el) sections.push({{ el, a }});
    }});
    const observer = new IntersectionObserver((entries) => {{
      entries.forEach(entry => {{
        if (entry.isIntersecting) {{
          tocLinks.forEach(a => a.classList.remove('active'));
          const hit = sections.find(s => s.el === entry.target);
          if (hit) hit.a.classList.add('active');
        }}
      }});
    }}, {{ threshold: 0.08, rootMargin: '-72px 0px -62% 0px' }});
    sections.forEach(s => observer.observe(s.el));
  </script>
</body>
</html>
"""


def build(source: Path, output_dir: Path, asset_dir: Path) -> Path:
    source_text = source.read_text(encoding="utf-8")
    protected, placeholders = protect_math(source_text)
    md = MarkdownIt("default", {"html": False, "linkify": False, "typographer": False})
    rendered = restore_math(md.render(protected), placeholders)
    title, description, article, toc_html = rewrite_soup(rendered)

    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / "index.html"
    html_path.write_text(render_html(title, description, article, toc_html), encoding="utf-8")

    target_asset_dir = output_dir / "assets/kimi_series"
    target_asset_dir.mkdir(parents=True, exist_ok=True)
    for png in sorted(asset_dir.glob("*.png")):
        shutil.copy2(png, target_asset_dir / png.name)
    return html_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--asset-dir", type=Path, default=DEFAULT_ASSET_DIR)
    args = parser.parse_args()

    html_path = build(args.source, args.output_dir, args.asset_dir)
    print(html_path)


if __name__ == "__main__":
    main()
