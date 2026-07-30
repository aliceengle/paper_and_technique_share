#!/usr/bin/env python3
"""Build a cleaned Markdown and static HTML share card for the Kimi K3 core-tech report."""

from __future__ import annotations

import argparse
import html
import re
import shutil
import unicodedata
from pathlib import Path

from bs4 import BeautifulSoup
from markdown_it import MarkdownIt


TECHNIQUE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = TECHNIQUE_DIR / "kimi_k3_core_tech_kda_attnres_stable_moe_report_20260730.md"
DEFAULT_MD_OUTPUT = TECHNIQUE_DIR / "kimi_k3_core_tech_share_card_20260730.md"
DEFAULT_HTML_OUTPUT_DIR = TECHNIQUE_DIR / "html/kimi-k3-core-tech-share-card-260730"


def slice_share_content(markdown: str) -> str:
    lines = markdown.splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == "## 0. 结论速览")
    end = next(
        (i for i, line in enumerate(lines) if line.strip() == "## 7. 待继续验证"),
        len(lines),
    )
    content = "\n".join(lines[start:end]).strip() + "\n"
    replacements = {
        "## 5. 对本仓库 vLLM / 推理实验的落点": "## 5. 工程落地与推理实验关注点",
        "对本仓库来说": "对推理服务落地来说",
        "本仓库": "推理服务实验",
        "### 2.6 AttnRes 的收益和待验证点": "### 2.6 AttnRes 的收益和边界",
        "待验证点": "边界",
        "风险或待验证点": "风险或边界",
    }
    for src, dst in replacements.items():
        content = content.replace(src, dst)
    return content


def protect_math(markdown: str) -> tuple[str, list[str]]:
    placeholders: list[str] = []

    def marker(index: int) -> str:
        return f"@@KIMI_MATH_PLACEHOLDER_{index}@@"

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
            f"@@KIMI_MATH_PLACEHOLDER_{idx}@@",
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


def wrap_h2_cards(soup: BeautifulSoup) -> None:
    root = soup
    nodes = list(root.contents)
    new_nodes = []
    i = 0
    while i < len(nodes):
        node = nodes[i]
        if getattr(node, "name", None) == "h2":
            card = soup.new_tag("section")
            card["class"] = "share-section-card"
            card.append(node.extract())
            i += 1
            while i < len(nodes) and getattr(nodes[i], "name", None) != "h2":
                card.append(nodes[i].extract())
                i += 1
            new_nodes.append(card)
        else:
            new_nodes.append(node)
            i += 1
    root.clear()
    for node in new_nodes:
        root.append(node)


def rewrite_soup(body_html: str) -> tuple[str, str]:
    soup = BeautifulSoup(body_html, "html.parser")
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

    wrap_h2_cards(soup)
    toc_html = "\n".join(
        f'<li class="level-{level}"><a href="#{hid}">{html.escape(text)}</a></li>'
        for level, hid, text in toc_items
    )
    return str(soup), toc_html


def render_html(article: str, toc_html: str) -> str:
    title = "Kimi K3 核心技术分享卡：KDA、AttnRes、Stable MoE"
    description = "聚焦 Kimi K3 的 KDA、Attention Residuals 与 Stable LatentMoE 三项核心结构技术"
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
      --bg: #eef3f7;
      --panel: #ffffff;
      --panel-soft: #f7fafc;
      --ink: #14202b;
      --ink-2: #425160;
      --ink-3: #7c8997;
      --border: #dbe5ee;
      --accent: #0f6973;
      --accent-2: #426fb1;
      --accent-bg: rgba(15, 105, 115, .08);
      --warn-bg: #fff8e8;
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
      background:
        linear-gradient(180deg, #f9fbfd 0, var(--bg) 420px),
        var(--bg);
      font-family: var(--font-sans);
      line-height: 1.76;
      -webkit-font-smoothing: antialiased;
    }}
    a {{ color: var(--accent); text-decoration-color: rgba(15,105,115,.28); text-underline-offset: 3px; }}
    a:hover {{ text-decoration-color: var(--accent); }}
    .progress-bar {{ position: fixed; inset: 0 0 auto 0; height: 3px; z-index: 100; }}
    .progress-bar .fill {{ height: 100%; width: 0; background: linear-gradient(90deg, var(--accent), var(--accent-2)); }}
    .page {{ max-width: var(--measure); margin: 0 auto; padding: 48px 28px 76px; }}
    .hero {{
      padding: 28px 0 30px;
      border-bottom: 1px solid var(--border);
      margin-bottom: 24px;
    }}
    .eyebrow {{
      color: var(--accent);
      font: 600 .78rem/1.4 var(--font-mono);
      letter-spacing: .03em;
      text-transform: uppercase;
      margin-bottom: .6rem;
    }}
    h1, h2, h3, h4 {{ color: var(--ink); font-family: var(--font-serif); line-height: 1.34; }}
    h1 {{ max-width: 900px; margin: 0 0 .9rem; font-size: clamp(2rem, 4vw, 3.4rem); font-weight: 700; }}
    .lead {{ max-width: 820px; margin: 0; color: var(--ink-2); font-size: 1.04rem; }}
    .topline {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 18px;
    }}
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
    .content-grid {{
      display: grid;
      grid-template-columns: minmax(0, 1fr);
      gap: 20px;
    }}
    .share-section-card {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 24px;
      box-shadow: 0 14px 36px rgba(20, 32, 43, .06);
    }}
    .share-section-card + .share-section-card {{ margin-top: 20px; }}
    h2 {{ margin: 0 0 1rem; padding-bottom: .6rem; border-bottom: 1px solid var(--border); font-size: 1.45rem; }}
    h3 {{ margin: 1.8rem 0 .65rem; font-size: 1.08rem; }}
    h4 {{ margin: 1.3rem 0 .45rem; font-size: 1rem; }}
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
    .table-wrap {{ overflow-x: auto; margin: 1rem 0 1.35rem; border: 1px solid var(--border); border-radius: 8px; }}
    table {{ width: 100%; min-width: 680px; border-collapse: collapse; font-size: .91rem; }}
    th, td {{ padding: .64rem .72rem; border-bottom: 1px solid var(--border); vertical-align: top; }}
    th {{ background: var(--panel-soft); color: var(--ink); font-weight: 600; text-align: left; }}
    tr:last-child td {{ border-bottom: 0; }}
    img {{
      display: block;
      max-width: 100%;
      margin: 1rem auto 1.25rem;
      border: 1px solid var(--border);
      border-radius: 10px;
      background: #fff;
    }}
    .mermaid {{
      margin: 1rem 0 1.35rem;
      padding: 1rem;
      background: var(--panel-soft);
      border: 1px solid var(--border);
      border-radius: 10px;
      overflow-x: auto;
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
      background: rgba(255, 255, 255, .88);
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
      .share-section-card {{ padding: 18px; border-radius: 10px; }}
      table {{ min-width: 620px; }}
    }}
    @media print {{
      .progress-bar, .toc {{ display: none !important; }}
      body {{ background: #fff; }}
      .page {{ max-width: none; padding: 0; }}
      .share-section-card {{ box-shadow: none; break-inside: avoid; }}
    }}
  </style>
</head>
<body>
  <div class="progress-bar"><div class="fill" id="progressFill"></div></div>
  <main class="page">
    <header class="hero">
      <div class="eyebrow">Kimi K3 Core Tech Share Card</div>
      <h1>Kimi K3 三项核心技术</h1>
      <p class="lead">聚焦 KDA、Attention Residuals、Stable MoE：分别从序列长度、模型深度、专家宽度三个维度解释 K3 的结构跃迁。</p>
      <div class="topline">
        <span class="pill">KDA / 1M Context</span>
        <span class="pill">Block AttnRes</span>
        <span class="pill">Stable LatentMoE</span>
        <span class="pill">Quantile Balancing</span>
      </div>
    </header>
    <nav class="toc" aria-label="分享卡目录">
      <div class="toc-label">目录</div>
      <ul id="tocList">
        {toc_html}
      </ul>
    </nav>
    <article class="content-grid">
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


def copy_referenced_images(markdown: str, source_dir: Path, output_dir: Path) -> None:
    for match in re.finditer(r"!\[[^\]]*]\(([^)]+)\)", markdown):
        target = match.group(1)
        if target.startswith(("http://", "https://", "data:")):
            continue
        src = source_dir / target
        if not src.exists():
            continue
        dst = output_dir / target
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def build(source: Path, md_output: Path, html_output_dir: Path) -> Path:
    share_md = slice_share_content(source.read_text(encoding="utf-8"))
    md_output.write_text(share_md, encoding="utf-8")

    protected, placeholders = protect_math(share_md)
    rendered = MarkdownIt("default", {"html": False, "linkify": False, "typographer": False}).render(protected)
    rendered = restore_math(rendered, placeholders)
    article, toc_html = rewrite_soup(rendered)

    html_output_dir.mkdir(parents=True, exist_ok=True)
    html_path = html_output_dir / "index.html"
    html_path.write_text(render_html(article, toc_html), encoding="utf-8")
    copy_referenced_images(share_md, source.parent, html_output_dir)
    return html_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--md-output", type=Path, default=DEFAULT_MD_OUTPUT)
    parser.add_argument("--html-output-dir", type=Path, default=DEFAULT_HTML_OUTPUT_DIR)
    args = parser.parse_args()

    html_path = build(args.source, args.md_output, args.html_output_dir)
    print(args.md_output)
    print(html_path)


if __name__ == "__main__":
    main()
