#!/usr/bin/env python3
"""Build the static HTML page for the five-paper KV-cache report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import build_domino_technique_html as base


TECHNIQUE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    TECHNIQUE_DIR
    / "md/KVcache/analysis/three_kv_cache_compression_directions_five_papers_analysis_20260910.md"
)
DEFAULT_OUTPUT_DIR = (
    TECHNIQUE_DIR / "html/kv-cache-compression-three-directions-five-papers-260910"
)

OLD_DESCRIPTION = (
    "Domino 技术分析：自回归 Drafting、因果建模解耦、训练目标与推理部署"
)
DESCRIPTION = (
    "KV Cache 三方向五论文分析：OSCAR、KVarN、VecInfer、NOVA-KV 与 "
    "RestoreKV 的机制、效果和遗留问题"
)
OLD_PILLS = """        <span class="pill">Domino</span>
        <span class="pill">自回归 Drafting</span>
        <span class="pill">因果解耦</span>
        <span class="pill">推理部署</span>"""
PILLS = """        <span class="pill">KV Cache</span>
        <span class="pill">标量量化</span>
        <span class="pill">向量量化</span>
        <span class="pill">Token 恢复</span>"""


def customize_page(html_text: str) -> str:
    replacements = {
        OLD_DESCRIPTION: DESCRIPTION,
        "Speculative Decoding / GLM-5.2": "KV Cache Compression / Five Papers",
        OLD_PILLS: PILLS,
    }
    for old, new in replacements.items():
        if old not in html_text:
            raise RuntimeError(f"base template marker missing: {old}")
        html_text = html_text.replace(old, new)
    return html_text


def build(source: Path, output_dir: Path) -> Path:
    html_path = base.build(source, output_dir)
    page_html = html_path.read_text(encoding="utf-8")
    page_html = customize_page(page_html)
    html_path.write_text(page_html, encoding="utf-8")
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
