# Technique 报告发布说明

本目录用于维护 Kimi / speculative decoding 相关技术报告、静态 HTML 页面和发布脚本。141 Windows 主工程中的目录用于编辑和生成内容；真正推送到 GitHub Pages 的远端仓库是单独的 `paper_and_technique_share` 仓库。

## 目录与远端关系

| 角色 | 机器 / 地址 | 路径 / 仓库 | 用途 |
|---|---|---|---|
| 207.169 影子工程 | `192.168.207.169` | `/nfs/3D/zhangleichao/zhangleichao/edge10_ws/anwsome_vllm_infer_code/contexts/kimi_k2_7/Technique` | Linux / 容器侧查看、准备材料的位置，不作为最终发布源 |
| 141 Windows 主工程 | `192.168.28.141` | `D:\claude_code_ws\anwsome_vllm_infer_code\contexts\kimi_k2_7\Technique` | 主工程中的 Technique 内容，生成 Markdown 对应 HTML |
| 141 发布仓库 | `192.168.28.141` | `D:\claude_code_ws\paper_and_technique_share_publish\Technique` | `Technique` 对应的 GitHub 发布仓库工作区 |
| Git 远端 | GitHub | `git@github.com:aliceengle/paper_and_technique_share.git` | GitHub Pages 内容源仓库 |
| Pages 入口 | GitHub Pages | `https://aliceengle.github.io/paper_and_technique_share/Technique/` | 线上报告入口 |

注意：`contexts\kimi_k2_7\Technique` 对应的线上发布远端不是 `aliceengle/anwsome_vllm_infer_code`，而是 `aliceengle/paper_and_technique_share`。不要把这个目录的 Pages 内容推到 `anwsome_vllm_infer_code` 的 `gh-pages` 分支。

## 凭据记录要求

- 141 SSH 登录用户：`admin`
- 141 SSH 地址：`192.168.28.141`
- 207 侧地址：`192.168.207.169`
- 密码：不要写入本 README，不要提交到 Git，不要进入 shell history。

密码应通过本机私有环境变量或受控密码管理工具注入，例如在 `192.168.207.169` 的临时 shell 中设置：

```bash
export WIN141_PASSWORD='<从受控渠道获取的 141 admin 密码>'
```

如果必须交接明文口令，只能放在团队约定的私有凭据渠道，不能提交到远程仓库。原因是本目录会推送到 GitHub，明文密码一旦进入 Git 历史就很难彻底移除。

## 从 207.169 登录 141

在 `192.168.207.169` 上执行：

```bash
SSH_OPTS="-o PreferredAuthentications=password -o PubkeyAuthentication=no -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
SSHPASS="$WIN141_PASSWORD" sshpass -e ssh $SSH_OPTS admin@192.168.28.141
```

只做一次连通性和路径检查时，可以从 207.169 直接发远程命令：

```bash
SSHPASS="$WIN141_PASSWORD" sshpass -e ssh $SSH_OPTS admin@192.168.28.141 \
  'cmd /c "hostname && cd /d D:\claude_code_ws\paper_and_technique_share_publish && git status --short --branch"'
```

预期主机名为 `IF-20240407WIBC`。

## 常规发布流程

以下操作都在 `192.168.28.141` 上执行。建议先 SSH 登录到 141，再运行命令，避免 Windows 引号在远程单行命令里出错。

### 1. 从主工程生成 HTML

以 speculative decoding 报告为例，先把 DeepSeek 技术报告复制到主工程的 Technique 目录，再生成 HTML：

```cmd
copy /Y D:\claude_code_ws\anwsome_vllm_infer_code\DeepSeek_technique\Report\technique\speculative_decoding_technique_comparison_glm52_report_20260721.md D:\claude_code_ws\anwsome_vllm_infer_code\contexts\kimi_k2_7\Technique\speculative_decoding_technique_comparison_glm52_report_20260721.md
python D:\claude_code_ws\anwsome_vllm_infer_code\contexts\kimi_k2_7\Technique\tools\build_speculative_decoding_html.py
```

生成结果：

```text
D:\claude_code_ws\anwsome_vllm_infer_code\contexts\kimi_k2_7\Technique\html\speculative-decoding-technique-comparison-glm52-260721\index.html
```

### 2. 同步到发布仓库

```cmd
copy /Y D:\claude_code_ws\anwsome_vllm_infer_code\contexts\kimi_k2_7\Technique\speculative_decoding_technique_comparison_glm52_report_20260721.md D:\claude_code_ws\paper_and_technique_share_publish\Technique\speculative_decoding_technique_comparison_glm52_report_20260721.md
xcopy /E /I /Y D:\claude_code_ws\anwsome_vllm_infer_code\contexts\kimi_k2_7\Technique\html\speculative-decoding-technique-comparison-glm52-260721 D:\claude_code_ws\paper_and_technique_share_publish\Technique\html\speculative-decoding-technique-comparison-glm52-260721
copy /Y D:\claude_code_ws\anwsome_vllm_infer_code\contexts\kimi_k2_7\Technique\tools\build_speculative_decoding_html.py D:\claude_code_ws\paper_and_technique_share_publish\Technique\tools\build_speculative_decoding_html.py
```

### 3. 检查页面结构

```cmd
find /c "<table" D:\claude_code_ws\paper_and_technique_share_publish\Technique\html\speculative-decoding-technique-comparison-glm52-260721\index.html
find /c "table-wrap" D:\claude_code_ws\paper_and_technique_share_publish\Technique\html\speculative-decoding-technique-comparison-glm52-260721\index.html
find /c "border-collapse: collapse" D:\claude_code_ws\paper_and_technique_share_publish\Technique\html\speculative-decoding-technique-comparison-glm52-260721\index.html
find /c ".mermaid svg" D:\claude_code_ws\paper_and_technique_share_publish\Technique\html\speculative-decoding-technique-comparison-glm52-260721\index.html
```

表格必须满足：

| 检查项 | 期望 |
|---|---|
| `<table` | 大于 0，数量随报告内容变化 |
| `table-wrap` | 至少等于表格数量，另加 CSS 选择器出现次数 |
| `border-collapse: collapse` | 存在 |
| `overflow-x: auto` | 存在，保证宽表横向滚动 |

### 4. 提交并推送

`kimi` 是日常工作分支；当前 GitHub Pages 实际从 `main` 服务，所以需要把同一提交推到 `kimi` 和 `main`。

```cmd
git -C D:\claude_code_ws\paper_and_technique_share_publish status --short --branch
git -C D:\claude_code_ws\paper_and_technique_share_publish diff --check
git -C D:\claude_code_ws\paper_and_technique_share_publish add Technique\speculative_decoding_technique_comparison_glm52_report_20260721.md Technique\html\speculative-decoding-technique-comparison-glm52-260721\index.html Technique\tools\build_speculative_decoding_html.py Technique\README.md
git -C D:\claude_code_ws\paper_and_technique_share_publish commit -m "update technique report docs"
git -C D:\claude_code_ws\paper_and_technique_share_publish push origin kimi
git -C D:\claude_code_ws\paper_and_technique_share_publish push origin kimi:main
```

如果只是 README 变更，只 add `Technique\README.md` 即可。

### 5. 验证线上 Pages

```cmd
curl.exe --ssl-no-revoke -L -I -H "Cache-Control: no-cache" --max-time 30 https://aliceengle.github.io/paper_and_technique_share/Technique/
curl.exe --ssl-no-revoke -L -I -H "Cache-Control: no-cache" --max-time 30 https://aliceengle.github.io/paper_and_technique_share/Technique/html/speculative-decoding-technique-comparison-glm52-260721/
```

报告页链接：

```text
https://aliceengle.github.io/paper_and_technique_share/Technique/html/speculative-decoding-technique-comparison-glm52-260721/
```

## 注意事项

- 所有发布操作以 `192.168.28.141` 上的 Windows 工作区为准，不直接从 `192.168.207.169` 的影子工程推送。
- `paper_and_technique_share_publish` 才是 `Technique` 的发布仓库；不要误用 `anwsome_vllm_infer_code-gh-pages`。
- 推送 Pages 时必须确认 `main` 已更新；只推 `kimi` 时，线上页面可能不会刷新。
- Windows OpenSSH 下复杂引号容易出错，长流程建议先 SSH 登录 141 再运行本地 `cmd` 命令。
- 运行 `curl.exe` 访问 GitHub Pages 时，如果 Windows Schannel 报证书吊销检查失败，可以加 `--ssl-no-revoke`。
- 生成 HTML 后必须检查表格样式，尤其是 `table-wrap`、`border-collapse: collapse`、`overflow-x: auto`。
- Mermaid 图居中依赖 `build_speculative_decoding_html.py` 中的 `.mermaid svg { margin: 0 auto; }`，不要手工只改生成后的 HTML 而忘记同步脚本。
- 不要把密码、token、SSH 私钥、GitHub PAT 写进 README 或提交历史。
