# Kimi 系列技术演进 HTML 分享页

本目录是 `kimi_series_technology_timeline_20260724.md` 生成的静态 HTML 包，形态与 `slchenchn.github.io/reports/FP4-survey-2606/fp4_post-train_doc.html` 类似：一个 `index.html` 加相对路径图片资产目录。

## 重新生成

```bash
python3 contexts/kimi_k2_7/Technique/tools/build_kimi_timeline_html.py
```

生成内容：

| 路径 | 用途 |
|---|---|
| `index.html` | 可分享的单页 HTML |
| `assets/kimi_series/*.png` | 论文与技术报告截图资产 |

## 本地预览

直接打开：

```text
contexts/kimi_k2_7/Technique/html/kimi-series-technology-timeline-2607/index.html
```

或启动临时静态服务：

```bash
cd contexts/kimi_k2_7/Technique/html
python3 -m http.server 8080
```

访问：

```text
http://127.0.0.1:8080/kimi-series-technology-timeline-2607/
```

## 发布为分享链接

如果使用 GitHub Pages，建议把本目录复制到 Pages 发布仓库或发布分支的：

```text
reports/kimi-series-technology-timeline-2607/
```

对应 URL 形态为：

```text
https://<github-user>.github.io/<repo>/reports/kimi-series-technology-timeline-2607/
```

当前仓库的 `https://aliceengle.github.io/anwsome_vllm_infer_code/` 访问结果为 404，说明尚未配置可用的 GitHub Pages 入口。
