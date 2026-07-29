# Kimi 系列技术路线研究、时间线重绘与结果文档写作任务书

> 日期：2026-07-29  
> 目标文档：`contexts/kimi_k2_7/Technique/kimi_series_technology_timeline_20260724.md`  
> 任务类型：Kimi 系列技术路线研究、结果文档大纲约束、互联网资料核验、141 主工程同步  
> 参考同步文档：`contexts/207_169_shadow_to_141_main_sync.md`  
> 当前任务状态：结果文档第三轮证据增强已完成并已同步 141；后续进入更细粒度截图裁剪、附录表格摘录和本仓库实验设计扩展

## 1. 任务目标

本任务不是只按给定知乎文章改写时间线，而是要系统研究 Kimi 系列从 K2 到 K3 的技术路线，并形成一份可持续补充的技术演进结果文档。研究来源包括用户给定知乎文章、知乎专栏系列、每个模型的官方博客、技术报告、模型卡、论文、开源仓库、benchmark 说明、相关技术论文和互联网上的高质量分析文章。

| 层级 | 目标 | 固定要求 |
|---|---|---|
| 技术路线研究 | 还原 Kimi 系列模型的技术演进 | 不能只引用知乎；必须结合 Kimi 官方技术报告、论文、模型卡、博客、GitHub、benchmark 和相关技术论文 |
| 时间线重绘 | 参照知乎文章 `2061492527589494896` 重绘 Kimi 时间线 | 时间线图以该文章的时间线为主；现有 md 中时间线里的冗余 benchmark、重复节点和无法支撑主线的信息要清理 |
| 时间线总表 | 在时间线图后给出总表 | 总表覆盖日期、模型或阶段、技术变化、能力变化、来源、证据状态，便于后续持续补证 |
| 三线深挖 | 参照知乎文章 `2061491373937791328` 和专栏 `c_2061491184288245579` 拆成三大技术章节 | 三章固定为“推理效率”“RL 后训练”“Agent 编排”；每章开头必须给技术点发展脉络图 |
| 技术点研究 | 每个技术点都单独建小节 | 每个小节必须写理论背景、原理机制、Kimi 落点、效果证据、演进关系、参考链接；必要时加入论文或网页截图 |
| Kimi-K4 预测 | 在结果文档第四章预测下一代可能创新 | 从推理效率、RL 后训练、Agent 编排三个角度分别给出可能方向、依据和风险 |
| 总结提炼 | 最后一章总结 Kimi 系列最重要创新 | 总结要回答“Kimi 最核心的技术路线是什么”“哪些创新真正影响能力和服务成本”“哪些仍需实验验证” |

最终文档不是简单堆链接，而是回答：Kimi 系列每一代为什么出现、技术演进解决了什么瓶颈、创新在哪里、理论依据是什么、工程效果如何、与相近技术相比差别是什么、后续 Kimi-K4 可能沿哪些方向继续演进、对本仓库 vLLM / disaggregated inference 实验有什么启发。

## 2. 输入资料与访问状态

资料搜集不能只围绕已给出的知乎链接。每个模型、每个技术点都必须建立“主来源 + 交叉验证来源 + 相近技术来源”的证据链。

| 来源 | 用途 | 当前访问状态 | 后续处理 |
|---|---|---|---|
| https://zhuanlan.zhihu.com/p/2061492527589494896 | Kimi 时间线主来源，即专栏第二章“Kimi 技术演进路线图与核心创新” | 2026-07-29 复核：141 命令行直连单篇返回 403；Jina Reader 只返回安全验证；但 141 Google Chrome 可打开全文，标题为“第二章：Kimi 技术演进路线图与核心创新”，正文长度约 18.9K 字符 | 可通过 141 Google Chrome 读取正文并校准时间线；写入结果文档时仍需用官方资料交叉验证关键事实 |
| https://zhuanlan.zhihu.com/p/2061491373937791328 | 系列第一章与三条扩展轴背景来源 | 2026-07-29 复核：141 命令行直连单篇返回 403；Jina Reader 可读正文；141 Google Chrome 可打开正文，标题为“第一章：Kimi 与 DeepSeek 大模型技术路线图：初学者系统学习指南” | 可作为已可读来源使用；仍需用官方资料交叉验证其中涉及 Kimi 技术事实 |
| https://zhuanlan.zhihu.com/c_2061491184288245579 | 系列专栏来源 | 2026-07-29 复核：141 直连专栏页 200，标题为“大模型初探”；Jina Reader 可读 5 篇目录和摘要；141 Google Chrome 可打开专栏页 | 作为系列文章入口；其余文章可继续通过 141 Google Chrome 或 Jina Reader 逐篇读取 |
| Kimi 官方博客 | 每代模型发布、能力定位、限制说明、评测条件 | 现有文档已引用部分 | 作为模型发布时间、能力变化和评测口径的首选来源 |
| Hugging Face 模型卡 | 模型架构规格、上下文长度、量化格式、使用限制 | 现有文档已引用部分 | 每个模型必须核验一遍，抽取结构化规格写入总表 |
| Kimi 技术报告和论文 | 架构、训练、后训练、推理系统的技术细节 | 需要逐篇阅读 | 每个关键技术点优先引用技术报告或论文，而不是只引用博客 |
| Kimi / Moonshot GitHub 仓库 | Kimi-K2、Kimi-K3、Kimi Linear、Mooncake 等实现线索 | 可继续联网搜索 | 用于补充配置、实现细节、README 中的效果数据和工程限制 |
| vLLM、Mooncake、相关推理系统论文 | 推理效率和服务化背景 | 可继续联网搜索 | 用于深挖 prefix cache、P/D 分离、KDA cache、低精度部署 |
| 相近技术论文 | 与 Kimi 技术做边界对比 | 需要按技术点搜集 | 包括 MoE、MLA、MQA/GQA、linear attention、DeltaNet、YaRN、QAT、RLHF/RLAIF、tool-use RL、multi-agent orchestration 等 |
| 高质量技术文章 | 辅助理解理论、原理和工程效果 | 需要筛选 | 可引用但不能替代官方或论文事实；每条引用要标明来源可信度 |

访问记录说明：141 主机为 `192.168.28.141`，主工程路径为 `D:\claude_code_ws\anwsome_vllm_infer_code`。本任务书已按同步文档要求同步到 141；后续正文资料抓取也优先在 141 上执行。

### 2.0 141 访问复核记录

2026-07-29 按用户给定的 141 登录信息执行复核：

```bash
sshpass -p 'admin' ssh \
  -o PreferredAuthentications=password \
  -o PubkeyAuthentication=no \
  -o StrictHostKeyChecking=no \
  -o UserKnownHostsFile=/dev/null \
  admin@192.168.28.141
```

复核结论：

| URL | 141 直连 | 141 + Jina Reader | 结论 |
|---|---|---|---|
| `https://zhuanlan.zhihu.com/p/2061492527589494896` | 命令行 `403`；Google Chrome 可读，标题为“第二章：Kimi 技术演进路线图与核心创新”，正文长度约 18.9K 字符 | 只返回“安全验证 / 请登录” | 第二章可通过 141 Google Chrome 读取；命令行和 Jina Reader 不能作为该文全文抓取方式 |
| `https://zhuanlan.zhihu.com/p/2061491373937791328` | 命令行 `403`；Google Chrome 可读 | 可读正文，长度约 20K 字符 | 第一章可作为可读来源 |
| `https://zhuanlan.zhihu.com/c_2061491184288245579` | `200`，SSR 页面长度约 58K 字符；Google Chrome 可读 | 可读专栏目录，包含 5 篇文章标题、链接和摘要 | 专栏目录可作为入口 |

专栏目录中已确认 5 篇文章：

| 章节 | 标题 | 链接 |
|---|---|---|
| 第一章 | Kimi 与 DeepSeek 大模型技术路线图：初学者系统学习指南 | `https://zhuanlan.zhihu.com/p/2061491373937791328` |
| 第二章 | Kimi 技术演进路线图与核心创新 | `https://zhuanlan.zhihu.com/p/2061492527589494896` |
| 第三章 | DeepSeek 技术演进路线图与核心创新 | `https://zhuanlan.zhihu.com/p/2061493772735849911` |
| 第四章 | 横向对比、技术互鉴与行业影响 | `https://zhuanlan.zhihu.com/p/2061494165410780782` |
| 第五章 | 初学者学习路径 | `https://zhuanlan.zhihu.com/p/2061494392616136722` |

### 2.1 每代模型资料清单

后续研究至少覆盖以下模型或技术节点。每个节点要收集发布时间、官方来源、模型规格、训练或后训练变化、推理系统变化、agent 能力变化、效果数据和限制说明。

| 节点 | 必查资料 | 重点抽取内容 |
|---|---|---|
| Kimi K2 Instruct | HF 模型卡、K2 技术主页、K2 技术报告、GitHub | 1T MoE、32B 激活、128K context、MuonClip、agentic coding、tool use 基线 |
| Kimi K2 Instruct 0905 | HF 模型卡、发布说明 | 256K context、coding 与 frontend coding 强化、相对 K2 的变化 |
| Kimi Linear / KDA 前置 | Kimi Linear GitHub、论文、相关 DeltaNet 论文 | KDA 理论、状态更新机制、KV cache 降低、1M context 和吞吐效果 |
| Kimi K2 Thinking | HF 模型卡、官方博客、技术页 | thinking、动态工具调用、INT4 QAT、连续 tool call、preserve thinking 的前置形态 |
| Kimi K2.5 | HF 模型卡、官方博客、arXiv 或技术报告 | 原生多模态、MoonViT、joint text-vision pre-training、zero-vision SFT、joint text-vision RL、Agent Swarm |
| Kimi K2.6 | HF 模型卡、官方博客、评测说明 | long-horizon coding、proactive autonomous execution、300 sub-agents、4000 steps、Swarm 扩展效果 |
| Kimi K2.7-Code | HF 模型卡、API 说明、benchmark 页面 | token-efficient thinking、真实世界长程 coding、MCP 能力、thinking token 降低 |
| Kimi K3 | 官方博客、HF 模型卡、K3 技术报告、GitHub | 2.8T、104B 激活、1M context、KDA、Stable LatentMoE、Quantile Balancing、AttnRes、Gated MLA、SiTU-GLU、MXFP4/MXFP8、MoonViT-V2 |
| Kimi Code / Kimi Work / API | 产品页、API 文档、benchmark harness 说明 | agent harness、工具协议、preserve thinking history、权限与执行环境 |

### 2.2 每个技术点的资料下限

| 资料类型 | 最低要求 |
|---|---|
| Kimi 内部来源 | 至少 1 个官方博客、模型卡、技术报告、论文或 GitHub 链接 |
| 理论来源 | 至少 1 个相近技术论文或经典论文，用来解释原理和差异 |
| 效果来源 | 至少 1 个 benchmark、消融、官方指标、README 效果表或可复现实验线索 |
| 网文来源 | 可引用知乎、博客、社区文章，但只能作为理解和观点来源，关键事实要交叉验证 |
| 截图来源 | 如果使用截图，必须写清楚截图来自哪篇论文、哪篇文章、哪张图或哪段表格 |

## 3. 目标文档结构

重构后的 `kimi_series_technology_timeline_20260724.md` 是本任务的研究结果文档。该文档必须按下面大纲书写，不能只停留在“时间线 + 链接索引”。

| 顺序 | 章节 | 必写内容 | 图表要求 |
|---:|---|---|---|
| 0 | 摘要与资料边界 | 主结论、资料范围、知乎访问状态、证据分级、本文不确定性 | 可用一张证据来源表 |
| 1 | Kimi 系列时间线 | 按知乎文章 `2061492527589494896` 重绘 Kimi 时间线；清理无用节点 | Mermaid timeline |
| 2 | Kimi 系列时间表 | 汇总每个模型或技术节点的日期、模型规格、技术变化、效果和来源 | 总表必须可独立阅读 |
| 3 | 推理效率演进 | 说明每代模型在推理效率上的技术点、理论、原理、效果和参考链接 | 章首必须有“推理效率技术发展脉络图” |
| 4 | RL 后训练演进 | 说明每代模型在 SFT、RL、thinking、tool use、多模态 RL、coding RL 上的演进 | 章首必须有“RL 后训练技术发展脉络图” |
| 5 | Agent 编排演进 | 说明每代模型在动态工具调用、Swarm、MCP、harness、权限治理上的演进 | 章首必须有“Agent 编排技术发展脉络图” |
| 6 | Kimi-K4 可能创新点预测 | 从推理效率、RL 后训练、Agent 编排三个角度预测 K4 可能路线 | 给出 K4 技术假设表和三线预测图 |
| 7 | 总结 | 总结 Kimi 系列最重要创新、路线特点、对本仓库 serving 的启发和待验证问题 | 给出核心创新矩阵 |
| 8 | 外部链接索引 | 每个正文小节都要能回链到参考来源 | 按官方、论文、知乎、技术文章、benchmark、代码仓库分类 |

现有文档里可迁移的内容包括 Kimi 官方模型规格、K2 到 K3 的公开 benchmark、KDA / Kimi Linear / Mooncake / vLLM 等链接。需要清理的是时间线图中的过多分数、重复说明、与知乎时间线不一致的节点组织方式、以及无法核验却写成确定事实的段落。

### 3.1 章节内部固定格式

推理效率、RL 后训练、Agent 编排三章内部必须统一成“章首总览 + 技术点小节 + 本章小结”的结构。

| 位置 | 固定内容 |
|---|---|
| 章首 | 一张 Mermaid 发展脉络图，标明哪个技术点由哪个模型在什么时候提出，后续是否演进、融合、替代或遗弃 |
| 章首表 | 一张“模型-技术点映射表”，列出 K2、K2-0905、K2-Thinking、K2.5、K2.6、K2.7-Code、K3 对应技术 |
| 技术点小节 | 每个技术点单独成节，不能多个关键技术混在一段里 |
| 技术点效果 | 必须写效果证据，包括 benchmark、官方指标、系统指标、成本变化、token 变化、cache 变化或公开消融 |
| 技术点素材 | 可以插入论文截图、网页截图或自绘图；截图必须有来源说明和链接 |
| 小节结尾 | 每个小节必须给“参考链接”，至少列出主来源和辅助来源 |
| 本章小结 | 说明该章技术路线的主导方向、已被继承的技术、可能被替代的技术、仍需验证的点 |

### 3.2 截图和图示要求

结果文档允许引用论文或网页截图，但要控制边界，避免把截图当成正文替代品。

| 类型 | 要求 |
|---|---|
| 论文截图 | 优先截取公式、架构图、消融表或核心结果图；必须写明论文标题、图号或表号、链接 |
| 网页截图 | 只截取与技术点直接相关的短片段、表格或官方指标；必须写明页面标题和链接 |
| 自绘图 | 推荐用 Mermaid 重画流程、时序、依赖和演进关系，减少大段截图依赖 |
| 图片存放 | 若需要保存图片，建议放在 `contexts/kimi_k2_7/Technique/assets/kimi_series/`，文件名使用 `技术点_来源_日期.png` 这类可读命名 |
| Markdown 引用 | 图片下方必须写一句 caption，说明“来源、用途、与本文结论的关系” |

## 4. 时间线重绘要求

### 4.1 Mermaid 图规则

时间线图必须短、稳、可渲染：

| 规则 | 要求 |
|---|---|
| 主线来源 | Kimi 时间线按照知乎文章 `2061492527589494896` 的时间线顺序绘制 |
| 节点粒度 | 一个节点只表达一个模型发布、关键技术公开或架构阶段变化 |
| 节点内容 | 最多三行：模型或阶段、关键技术变化、能力定位 |
| 不放内容 | 不在 Mermaid 节点中堆完整 benchmark 表、不写长句、不写来源链接 |
| 特殊字符 | Mermaid 节点中避免直接写 `<`、`>`、`|`；必须出现时改成描述性文字 |
| 章节划分 | 优先按知乎文章给出的阶段划分；如果文章没有阶段名，再按 K2 基线、Thinking、多模态与 Swarm、K3 架构跳变等阶段组织 |

### 4.2 时间线总表字段

Mermaid 图后必须补一个总表，建议字段如下：

| 字段 | 说明 |
|---|---|
| 时间 | 年月日或月份；不确定时写“待复核” |
| 节点 | 模型、论文、技术博客、产品形态或系统能力 |
| 类型 | 模型发布、架构技术、后训练技术、推理系统、产品 harness、benchmark |
| 关键变化 | 相对上一节点的主要变化 |
| 技术关键词 | MoE、KDA、QAT、Agent Swarm 等 |
| 代表指标 | 只放最能说明变化的 1 到 3 个指标 |
| 主来源 | 知乎、官方博客、HF、GitHub、arXiv 等 |
| 证据状态 | 已核验、待 141 登录复核、需官方交叉验证 |

### 4.3 当前候选节点

以下只是根据现有文档和公开资料整理的候选，不替代知乎文章最终时间线。执行正文改造时，若知乎文章时间顺序、节点名称或阶段划分不同，以知乎文章为准，并在总表证据状态中记录差异。

| 候选时间 | 候选节点 | 可能保留的技术关键词 | 备注 |
|---|---|---|---|
| 2025-07 | Kimi K2 Instruct | 1T MoE、32B 激活、128K context、MuonClip、agentic coding | K2 基线 |
| 2025-09 | Kimi K2 Instruct 0905 | 256K context、agentic coding、frontend coding | 上下文和 coding 能力增强 |
| 2025-10 | Kimi Linear / KDA 前置 | KDA、1M context、KV cache 压缩、linear attention | 是否进入主时间线取决于知乎文章 |
| 2025-11 | Kimi K2 Thinking | thinking、动态工具调用、INT4 QAT、preserve thinking 前置 | 后训练和 agent 能力关键节点 |
| 2026-01 | Kimi K2.5 | 原生多模态、MoonViT、joint text-vision RL、Agent Swarm | 多模态 agent 阶段 |
| 2026-04 | Kimi K2.6 | long-horizon coding、proactive autonomous execution、300 sub-agents | Swarm 扩展阶段 |
| 2026-06 | Kimi K2.7-Code | token-efficient thinking、真实世界长程 coding、MCP | coding agent 强化 |
| 2026-07 | Kimi K3 | 2.8T、1M context、KDA、Stable LatentMoE、MXFP4/MXFP8 | 架构跳变 |

## 5. 三章技术深挖任务

正文技术深挖固定拆成三章：推理效率、RL 后训练、Agent 编排。每章必须先讲技术演进逻辑，再逐个技术点展开。每个技术点至少回答：问题背景、理论基础、原理机制、Kimi 中的落点、提出时间、后续演进或遗弃状态、效果证据、相近技术对比、工程影响、参考链接与待验证问题。

每章开头必须有一张发展脉络图，图中要标出：

| 图中元素 | 要求 |
|---|---|
| 模型节点 | K2、K2-0905、K2-Thinking、K2.5、K2.6、K2.7-Code、K3 |
| 技术节点 | 该章所有关键技术点 |
| 时间信息 | 技术首次出现或公开的时间 |
| 演进关系 | 继承、增强、融合、替代、弱化或待确认 |
| 结果线索 | 至少标注 1 到 2 个能说明该路线有效的效果指标 |

### 5.1 第一章：推理效率演进

本章主线是：从“稀疏激活扩大容量”到“长上下文降低 cache 成本”，再到“服务系统重构和低精度部署”。建议坑位如下：

| 技术坑位 | 相关模型或阶段 | 必写问题 | 重点创新与效果 |
|---|---|---|---|
| 稀疏 MoE 与专家路由 | K2 到 K3 | 为什么 K2.x 选择 1T/32B MoE，K3 为什么扩到 2.8T/104B | 总容量扩展与单 token 激活计算解耦；必须写 $r_{\text{active}}=k/N_{\text{experts}}$ 等基础公式和 K2/K3 激活比例对比 |
| Stable LatentMoE | K3 | K3 为什么需要稳定大专家池 | 在 896 experts 下提升路由稳定性和可服务性；说明是否替代普通 Top-k MoE 的部分训练假设 |
| Quantile Balancing | K3 | 专家负载不均如何影响训练和推理 | 用 router-score quantile 降低负载均衡启发式超参敏感性；必须对比 auxiliary loss / load balancing |
| MLA 与 Gated MLA | K2.5 到 K3 | 长上下文 KV cache 为什么成为瓶颈 | 用 latent 表示压缩 KV，Gated MLA 进一步提高选择性；必须解释与 MHA、MQA、GQA 的区别 |
| KDA / Kimi Delta Attention | Kimi Linear、K3 | K3 为什么引入 KDA，Kimi Linear 与 K3 是什么关系 | 用状态更新承载长程信息，面向 1M context 降低 cache 和解码成本；必须解释 DeltaNet、linear attention、有限状态 memory 的理论关系 |
| Attention Residuals | K3 | 深层模型如何避免信息随深度损失 | 跨深度选择性检索表示，而不是简单逐层累积；要说明和普通 residual connection 的差异 |
| YaRN、256K、1M context | K2、K2-0905、K3 | K2 到 K3 上下文长度如何演进 | 区分 RoPE 扩展、RAG、原生长上下文和 cache 系统；说明 256K 到 1M 的成本变化 |
| KDA prefix cache | K3、vLLM | KDA 为什么改变传统 prefix cache 语义 | 把 KDA 状态纳入 prefix 复用和 vLLM 支持范围；需要联系本仓库 vLLM 服务实验 |
| Mooncake / P-D 分离 | K3 API、Mooncake | 长程 coding 为什么适合 prefill/decode/cache 分离 | 通过长 prefix cache 复用提升 coding workload 经济性；说明 cache hit、TTFT、TPOT、调度复杂度 |
| QAT、INT4、MXFP4、MXFP8 | K2-Thinking、K2.7-Code、K3 | 低精度为什么要从后训练阶段介入 | 从 PTQ 转向 QAT，让服务格式进入训练目标；说明 INT4、MXFP4、MXFP8 的差异和效果 |
| token-efficient thinking | K2.7-Code、K3 | K2.7-Code 为什么强调思考 token 减少 | 在保持长程推理状态的同时压低推理成本；说明 token 成本、延迟、质量之间的 tradeoff |

本章要特别联系本仓库：`toy_proxy_server.py`、`mooncake_connector_proxy.py`、`disagg_proxy_demo.py` 和 vLLM serving 脚本中的 P/D 分离、cache、router、benchmark 场景，说明这些 Kimi 技术对本地实验的启发。

### 5.2 第二章：RL 后训练演进

本章主线是：从 instruction tuning 到 thinking，再到 tool use、long-horizon coding、多模态任务和 agentic RL。建议坑位如下：

| 技术坑位 | 相关模型或阶段 | 必写问题 | 重点创新与效果 |
|---|---|---|---|
| Instruction SFT 到 Thinking SFT | K2、K2-Thinking | 普通 SFT 为什么不足以支撑长任务 | 显式推理轨迹、长程状态和错误恢复进入训练目标；说明 thinking 数据构造和普通 instruction 数据差异 |
| 工具调用 RL | K2-Thinking、K2.5、K3 | function calling 与动态 tool use 的训练差异 | 奖励不只看最终答案，还要覆盖工具选择、参数、读取结果和恢复；必须解释多轮 tool-call credit assignment |
| preserve thinking 训练 | K2.7-Code、K3 | 为什么 K3 对历史 thinking 回传敏感 | 把推理历史变成会话状态的一部分，影响 harness 设计；说明历史裁剪错误为什么会伤质量 |
| long-horizon coding RL | K2.6、K2.7-Code、K3 | 为什么 coding benchmark 需要长程任务训练 | 训练模型跨文件修改、测试、修复和多轮验证；必须引用 coding benchmark 效果 |
| joint text-vision RL | K2.5、K3 | 多模态 agent 为什么不是简单接视觉 encoder | 视觉证据直接参与后训练任务和奖励；说明多模态 RL 与 OCR/外接视觉工具的差异 |
| zero-vision SFT | K2.5 | 为什么需要无视觉样本阶段 | 在引入视觉前保留文本推理和工具能力，减少能力坍缩；需要查证其训练顺序和目的 |
| reasoning effort | K3 | effort mode 如何影响成本、延迟和分数 | 把思考深度作为可调预算，而非固定输出风格；必须记录 benchmark 的 effort 口径 |
| token efficiency RL | K2.7-Code、K3 | 如何减少无效 thinking token | 奖励压缩推理过程，同时保留关键状态和可恢复性；写清质量和 token 成本的效果证据 |
| 自验证与执行反馈 | K2.6、K2.7-Code、K3 | coding agent 如何用测试结果自我修正 | 把执行结果、失败日志和二次修复纳入学习闭环；说明与普通 RLHF 的差异 |
| 多目标奖励与安全约束 | K3 及后续 | agent 模型如何同时优化能力、成本、稳定性和权限边界 | 结合 K3 limitations、system prompt 约束和 benchmark harness 讨论 |

每个坑位必须补至少一个可访问来源。知乎观点可作为主线，但技术事实要用 Kimi 官方、HF、GitHub、arXiv、benchmark 说明交叉验证。

### 5.3 第三章：Agent 编排演进

本章主线是：从单步工具调用到长程状态，再到多 agent 并行、产品 harness 和权限治理。建议坑位如下：

| 技术坑位 | 相关模型或阶段 | 必写问题 | 重点创新与效果 |
|---|---|---|---|
| 动态工具调用 | K2-Thinking、K3 | 为什么普通 function calling 不等于 agent | 多轮选择工具、读结果、修错和继续规划；说明连续 200 到 300 次 tool call 的意义 |
| Agent Swarm | K2.5、K2.6 | 多 sub-agent 如何提升复杂任务覆盖率 | 并行探索、任务拆分、预算分配和结果合并；说明从引入到 300 sub-agents 的演进 |
| 300 sub-agents 与 4000 steps | K2.6 | 大规模 Swarm 的调度瓶颈在哪里 | 长历史压缩、冲突消解、全局一致性；必须写清是否被 K2.7/K3 继承或融合 |
| Kimi Code harness | K2.7-Code、K3 | benchmark 分数为什么与 harness 强相关 | 工具协议、终端、文件、测试和 thinking history 共同决定结果；说明与裸模型 API 的差异 |
| MCP 与外部工具生态 | K2.7-Code、K3 | MCP Mark、MCP Atlas 这类评测说明了什么 | 从封闭工具调用转向标准化工具生态；说明协议标准化和工具发现 |
| Kimi Work | K2.5 到 K3 产品形态 | 知识工作 agent 与 coding agent 有何不同 | 文件、浏览、可视化 artifact、视频和 dashboard 的多工具协同 |
| 上下文管理 | K2-0905、K3 | 1M context 是否可以替代检索和摘要 | 长上下文、检索、摘要、cache 和权限边界需要组合；说明长上下文不是 RAG 替代品 |
| 主动执行与权限 | K2.6、K3 | proactive autonomous execution 的风险是什么 | 需要审批、预算、回滚和 AGENTS.md 行为约束；结合 K3 limitations 写 |
| Agent 评测体系 | K2-Thinking 到 K3 | SWE、Terminal、BrowseComp、Toolathlon、MCP 类评测各看什么 | 必须记录 effort、工具预算、上下文策略和 harness；说明跨模型分数不可直接比的条件 |
| 多 agent 失败恢复 | K2.6 以后 | sub-agent 冲突、重复工作、失败工具如何处理 | 要写编排层的失败检测、重新规划、结果仲裁和最终一致性 |

本章写作时要避免把 agent 讲成产品宣传。每个技术点都要落到编排机制、状态管理、工具协议、评估变量和工程风险。

## 6. 第四章：Kimi-K4 可能创新点预测

结果文档第四章必须基于前三章的技术演进，预测下一代 Kimi-K4 可能出现的创新点。预测不能写成空泛展望，必须给出依据、技术假设、可能收益、实现难点和验证方式。

### 6.1 K4 预测写作框架

| 角度 | 必写问题 | 可能方向示例 | 证据依据 |
|---|---|---|---|
| 推理效率 | K3 的 1M context、KDA、LatentMoE、MXFP4/MXFP8 之后，下一个效率瓶颈是什么 | 更强 KDA/linear attention 混合、动态稀疏激活、KV/cache 分层、跨会话 prefix reuse、speculative decoding、多级 P/D 分离、端侧或边缘低精度 | K3 技术报告、Kimi Linear、vLLM/Mooncake 进展、低精度和推理系统论文 |
| RL 后训练 | K3 的 thinking、tool use、long-horizon coding 之后，后训练还会怎么演进 | 更细粒度 process reward、自动课程学习、长程任务执行 RL、多模态工具 RL、token-efficiency reward、self-verification reward、环境反馈 RL | K2-Thinking、K2.5、K2.6、K2.7-Code、K3 的后训练线索和相关 RL 论文 |
| Agent 编排 | Swarm、MCP、Kimi Code/Kimi Work 之后，agent 系统可能怎么升级 | 层级化 multi-agent、持久化记忆、工具市场、权限沙箱、计划-执行-审计分离、跨任务知识复用、自动回滚和验证 | Agent Swarm、MCP benchmark、Kimi Code harness、K3 limitations、生产 agent 安全文章 |

### 6.2 K4 预测章节产物

| 产物 | 要求 |
|---|---|
| K4 技术假设表 | 每个假设列出所属角度、可能继承自哪个 Kimi 技术点、预期收益、实现难点、验证指标 |
| 三线预测图 | 用 Mermaid 画出推理效率、RL 后训练、Agent 编排三条路线从 K2 到 K3 再到 K4 的推演 |
| 风险和反例 | 每个预测都要写不确定性，例如“公开资料不足”“可能只是产品 harness 改进”“可能被成本约束推迟” |
| 对本仓库启发 | 写出哪些方向可转化为 vLLM serving、Mooncake/P-D 分离、benchmark 或 agent harness 实验 |

### 6.3 K4 预测边界

| 规则 | 要求 |
|---|---|
| 不能臆造事实 | K4 未发布内容只能写“可能”“推测”“技术假设”，不能写成已发生 |
| 必须有依据链 | 每个预测至少关联一个 K2 到 K3 已发生的技术趋势和一个外部技术趋势 |
| 避免泛化 | 不写“更强、更快、更智能”这类空话，必须落到具体技术机制 |
| 预测可验证 | 每个方向要给出未来可验证的指标，如 TTFT、TPOT、cache hit、token usage、SWE、Terminal-Bench、MCP benchmark 等 |

## 7. 单个技术点分析模板

后续正文每个技术坑位都按下面模板展开，避免只写概念堆砌：

| 小节 | 必填内容 |
|---|---|
| 背景问题 | 这项技术解决哪个规模、成本、长上下文、训练稳定性或 agent 编排问题 |
| 理论背景 | 相关论文或经典方法是什么，核心数学形式或理论假设是什么 |
| 技术定义 | 用 2 到 4 句话解释技术机制，必要时给公式、流程图或伪代码 |
| 原理机制 | 详细讲清输入、状态、计算路径、训练目标、推理路径或编排机制 |
| Kimi 落点 | 出现在 K2、K2.5、K2.6、K2.7-Code、K3 或 Kimi Linear 的哪个阶段，首次公开时间是什么 |
| 演进状态 | 后续是否被继承、增强、融合、替代、弱化或遗弃；证据是什么 |
| 创新点 | 相比前一代 Kimi 或相近技术，真正新的地方是什么 |
| 相近技术对比 | 与 Dense、Switch/GShard、MQA/GQA、RAG、PTQ、普通 function calling 等做边界区分 |
| 效果证据 | 写 benchmark、消融、系统指标、成本变化、token 降低、cache 降低、吞吐提升或官方声明 |
| 截图或图示 | 如有论文图、网页表格或官方图，插入截图或重画 Mermaid，并写 caption |
| 工程影响 | 对 vLLM、Mooncake、P/D 分离、cache、低精度、benchmark、agent harness 或本仓库脚本的影响 |
| 参考链接 | 小节末尾必须列出链接，至少包含主来源和辅助来源 |
| 待验证问题 | 仍需实验、源码、模型卡或 141 登录态补证的问题 |

如果某项技术只在知乎文章中出现、官方资料无法交叉验证，正文必须明确标注“知乎文章观点，待官方资料验证”，不能写成确定事实。

### 7.1 技术点小节推荐模板

每个技术点小节建议按下面小标题写：

```markdown
#### 技术名

**提出位置与演进状态**
说明该技术首次出现在 Kimi 哪一代、什么时候公开，后续在哪些模型中被继承、增强、融合或替代。

**理论背景**
解释相关论文、数学形式、经典问题和为什么需要这项技术。

**原理机制**
用公式、流程图或步骤说明机制，不只写结论。

**Kimi 中的实现和创新**
说明 Kimi 公开资料里到底做了什么，与前一代或相近技术相比新在哪里。

**效果与证据**
列出 benchmark、系统指标、消融表、官方数据或可复现实验线索。

**工程影响**
说明对 vLLM、Mooncake、P-D 分离、cache、量化、agent harness 或本仓库实验的影响。

**参考链接**
- 主来源：...
- 论文或技术背景：...
- 辅助文章：...

**待验证问题**
列出仍需源码、实验、141 登录态或官方资料确认的问题。
```

## 8. 互联网搜索与证据沉淀流程

后续资料搜索优先在 141 上执行，因为当前 207.169 直连知乎失败，而 141 至少可以访问专栏 SSR 页面。

### 8.1 141 连接参数

```bash
SSH_OPTS="-o PreferredAuthentications=password -o PubkeyAuthentication=no -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
sshpass -p 'admin' ssh $SSH_OPTS admin@192.168.28.141
```

### 8.2 知乎正文核验

| 步骤 | 操作 | 产物 |
|---:|---|---|
| 1 | 在 141 浏览器或可登录环境打开两个单篇和专栏页 | 记录标题、发布时间、作者、章节结构 |
| 2 | 导出或复制可引用正文片段 | 保存到本地临时资料，不把大段版权正文直接提交进仓库 |
| 3 | 抽取时间线节点、三章框架和作者观点 | 写入总表，标注来源为知乎 |
| 4 | 用官方资料交叉验证关键事实 | 对每个技术点补官方、论文、模型卡或 GitHub 链接 |
| 5 | 抽取可引用截图或需要重画的图表 | 记录图片来源，不把大段正文截图当作主要内容 |
| 6 | 更新目标 md | 只写归纳，不复制原文长段 |

### 8.3 推荐搜索关键词

| 主题 | 关键词 |
|---|---|
| Kimi 时间线 | `Kimi K2 K2.5 K2.6 K2.7 K3 timeline`、`Kimi K3 technical report`、`Kimi K2 Thinking` |
| 推理效率 | `Kimi Delta Attention`、`Kimi Linear KDA`、`KDA prefix cache vLLM`、`Mooncake disaggregated inference`、`MXFP4 MXFP8 QAT`、`Stable LatentMoE Quantile Balancing`、`Gated MLA Attention Residuals` |
| RL 后训练 | `Kimi K2 Thinking tool use RL`、`Kimi K2.5 joint text vision RL`、`long horizon coding RL agent`、`process reward model tool use`、`agentic coding reinforcement learning`、`reasoning effort token efficiency` |
| Agent 编排 | `Kimi Agent Swarm`、`Kimi Code harness`、`MCP Atlas Kimi`、`MCP Mark verified`、`preserve thinking history`、`multi agent orchestration LLM`、`tool use benchmark agent` |
| K4 预测 | `LLM inference roadmap 2026`、`agentic RL roadmap`、`multi agent memory tool ecosystem`、`long context inference cache future` |

### 8.4 证据状态标签

| 标签 | 含义 |
|---|---|
| 已核验 | 官方博客、HF、GitHub、论文或可访问网页能直接支撑 |
| 待 141 登录复核 | 只知道知乎链接或专栏入口，正文仍需登录态确认 |
| 需官方交叉验证 | 知乎有说法，但尚未找到官方或论文支撑 |
| 待实验验证 | 涉及本仓库 serving、cache、吞吐、benchmark 的工程结论 |
| 不纳入正文 | 无来源、与主题弱相关、无法解释技术演进的内容 |

## 9. 正文修改执行步骤

| 阶段 | 操作 | 验收标准 |
|---:|---|---|
| 1 | 备份阅读现有 md，列出现有章节与可复用段落 | 不覆盖用户已有有效内容；只做目标文档重构 |
| 2 | 建立资料清单 | 每个模型节点都有官方、模型卡、论文/报告或 GitHub 来源；每个技术点都有主来源和辅助来源 |
| 3 | 在 141 上核验知乎时间线和三章框架 | 至少记录两个单篇的标题、时间线节点、章节观点和专栏 5 篇文章目录 |
| 4 | 重绘 Kimi 时间线 | Mermaid 图能渲染，节点不臃肿，顺序与知乎文章一致 |
| 5 | 增加 Kimi 时间表 | 表格字段完整，证据状态清楚，可独立阅读 |
| 6 | 改写推理效率章 | 章首有发展脉络图；每个坑位有理论、原理、效果、Kimi 落点和参考链接 |
| 7 | 改写 RL 后训练章 | 章首有发展脉络图；区分 SFT、thinking、tool use、RL、effort、多模态后训练和 coding RL |
| 8 | 改写 Agent 编排章 | 章首有发展脉络图；区分模型能力、harness、工具协议、Swarm、权限治理和评测变量 |
| 9 | 新增 Kimi-K4 预测章 | 从推理效率、RL 后训练、Agent 编排三角度给出技术假设、依据、风险和验证指标 |
| 10 | 新增总结章 | 总结 Kimi 系列最重要创新、技术路线特点、本仓库启发和待验证问题 |
| 11 | 处理截图和图示 | 需要截图时保存到 assets，caption 和来源链接完整；优先用 Mermaid 自绘 |
| 12 | 清理外部链接索引 | 删除正文未引用链接，补齐每个小节实际引用链接 |
| 13 | Markdown 渲染检查 | 表格不破、Mermaid 特殊字符不破、无明显重复段落、每小节有参考链接 |
| 14 | 同步到 141 | 任务书和最终 md 均同步到主工程对应路径 |

## 10. 删除与保留准则

| 类型 | 处理方式 |
|---|---|
| Mermaid 时间线里的完整 benchmark 列表 | 删除或移到总表，只保留 1 个代表性指标 |
| 与知乎时间线不一致的阶段名 | 以知乎文章为准重命名，差异写进证据状态 |
| 无法访问的知乎正文结论 | 不直接写成事实，先标“待 141 登录复核” |
| 官方模型规格和技术报告事实 | 保留，但要移动到对应技术章或总表 |
| 技术点理论、原理和效果证据 | 必须保留并补充，不允许只写“技术名 + 一句话介绍” |
| 每个小节参考链接 | 必须保留；没有参考链接的小节不算完成 |
| 论文或网页截图 | 只保留与技术点直接相关的图、公式或表；必须带 caption 和来源 |
| 重复的链接清单 | 合并到“外部链接索引”，只留正文实际引用项 |
| 只描述产品营销、不解释技术机制的内容 | 删除或压缩为产品 harness 背景 |
| 与本仓库 vLLM serving 无关的泛泛 benchmark | 删除或只在对比表中保留必要指标 |

## 11. 141 同步要求

本任务书创建后必须同步到 141 主工程：

| 项 | 路径 |
|---|---|
| 207.169 影子工程文件 | `/nfs/3D/zhangleichao/zhangleichao/edge10_ws/anwsome_vllm_infer_code/contexts/kimi_k2_7/Technique/tasks/kimi_series_technology_timeline_rework_taskbook_20260729.md` |
| 141 主工程目标文件 | `D:\claude_code_ws\anwsome_vllm_infer_code\contexts\kimi_k2_7\Technique\tasks\kimi_series_technology_timeline_rework_taskbook_20260729.md` |

推荐同步命令：

```bash
SSH_OPTS="-o PreferredAuthentications=password -o PubkeyAuthentication=no -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
SHADOW_ROOT=/nfs/3D/zhangleichao/zhangleichao/edge10_ws/anwsome_vllm_infer_code
MAIN_ROOT='D:/claude_code_ws/anwsome_vllm_infer_code'

sshpass -p 'admin' ssh $SSH_OPTS admin@192.168.28.141 \
  'if not exist "D:\claude_code_ws\anwsome_vllm_infer_code\contexts\kimi_k2_7\Technique\tasks" mkdir "D:\claude_code_ws\anwsome_vllm_infer_code\contexts\kimi_k2_7\Technique\tasks"'

sshpass -p 'admin' scp $SSH_OPTS \
  "$SHADOW_ROOT/contexts/kimi_k2_7/Technique/tasks/kimi_series_technology_timeline_rework_taskbook_20260729.md" \
  admin@192.168.28.141:"$MAIN_ROOT/contexts/kimi_k2_7/Technique/tasks/kimi_series_technology_timeline_rework_taskbook_20260729.md"
```

## 12. 完成判定

本任务书阶段完成标准：

| 检查项 | 标准 |
|---|---|
| 本地任务书 | 文件存在于 `contexts/kimi_k2_7/Technique/tasks/` |
| 内容完整性 | 覆盖资料范围、结果文档大纲、时间线重绘、时间线总表、三章技术深挖、K4 预测、总结、技术坑位模板、截图要求、141 搜索与同步流程 |
| 141 同步 | 141 主工程对应路径存在同名文件 |
| 后续可执行性 | 任何接手者可按本文继续完成目标 md 重构 |

后续正文重构阶段完成标准：

| 检查项 | 标准 |
|---|---|
| 时间线 | 按知乎文章时间线重绘，图表可渲染 |
| 总表 | 时间、节点、类型、关键变化、技术关键词、来源和证据状态齐全 |
| 三章深挖 | 推理效率、RL 后训练、Agent 编排三章均完成，章首有发展脉络图，且每种技术都有独立分析 |
| 技术点质量 | 每个技术点有理论背景、原理机制、Kimi 落点、演进状态、效果证据、工程影响、参考链接和待验证问题 |
| K4 预测 | 从推理效率、RL 后训练、Agent 编排三个角度给出可能创新、依据链、风险和验证指标 |
| 总结 | 提炼 Kimi 系列最重要创新、技术路线和对本仓库的启发 |
| 来源 | 知乎观点与官方事实分清楚，不把未核验内容写成确定结论；每个小节都有参考链接 |
| 截图 | 若使用截图，图片路径、caption、来源链接完整 |
| 同步 | 修改后的目标 md 同步到 141 主工程 |

## 13. 执行记录

### 13.1 2026-07-29 首轮正文重构

| 时间 | 操作 | 结果 |
|---|---|---|
| 2026-07-29 12:23 CST | 重构 `contexts/kimi_k2_7/Technique/kimi_series_technology_timeline_20260724.md` | 新版文档 1412 行，约 110KB，按“摘要与资料边界、Kimi 时间线、时间表、推理效率、RL 后训练、Agent 编排、K4 预测、总结、外部链接索引”重写 |
| 2026-07-29 12:24 CST | Markdown 结构检查 | 表格列数检查通过；代码块闭合；包含 5 个 Mermaid 图；32 个技术点参考链接区；32 个待验证问题区 |
| 2026-07-29 12:25 CST | 敏感信息检查 | 目标文档和任务书均未写入知乎账号、密码或临时登录变量 |
| 2026-07-29 12:26 CST | 同步到 141 主工程 | 任务书和目标文档均同步到 `D:\claude_code_ws\anwsome_vllm_infer_code` |
| 2026-07-29 12:27 CST | 141 同步一致性验证 | 目标文档 SHA256 为 `4f087a8f94ce76acced20509b492089febc491257e702b5e24d492fdda24b228`；任务书补写执行记录后已再次同步，最终一致性以同步命令输出为准 |

### 13.2 本轮已覆盖内容

| 模块 | 覆盖状态 |
|---|---|
| 知乎时间线 | 已按第二章四阶段重绘，从 2023-10 Kimi Chat 到 2026-07 K3 |
| 时间线总表 | 已加入日期、节点、类型、规格、关键变化、技术关键词、代表指标、来源和证据状态 |
| 推理效率章 | 已覆盖 MoE、Stable LatentMoE、Quantile Balancing、MLA/Gated MLA、KDA、MoBA、AttnRes、长上下文/cache、Mooncake、checkpoint-engine、QAT、MuonClip、token-efficient thinking |
| RL 后训练章 | 已覆盖 K1.5 long2short、K2 tool-use/self-critique、Thinking SFT、工具 RL、多模态 RL、PARL、long-horizon coding RL、preserve thinking、reasoning effort、多目标安全约束 |
| Agent 编排章 | 已覆盖动态工具调用、长工具链状态管理、Agent Swarm、300 sub-agents/4000 steps、Kimi Code、MCP、Kimi Work、上下文管理、主动执行权限、Agent 评测 |
| K4 预测 | 已从推理效率、RL 后训练、Agent 编排三条线给出技术假设、依据、收益、难点、指标和风险 |
| 本仓库启发 | 已关联 vLLM serving、P-D 分离、Mooncake connector、proxy/router、benchmark 和 agent harness |

### 13.3 后续扩展清单

| 优先级 | 待办 | 目标 |
|---|---|---|
| P0 | 逐页阅读 K2 technical report 和 K3 technical report | 补齐 MoE scaling、MuonClip、MLA、Stable LatentMoE、Quantile Balancing、AttnRes 的原始公式、图号和消融 |
| P0 | 读取 Kimi Linear 完整论文 | 已完成首轮深读并入正文；已补 KDA 状态更新、3:1 hybrid、KV cache 降低、1M context decode/prefill 速度和 4 张截图；后续可继续裁剪表格 |
| P1 | 读取 K2.5 arXiv 与 PARL 相关段落 | 已完成首轮深读并入正文；已补 Agent Swarm/PARL 公式、critical steps、runtime、BrowseComp/WideSearch 设置和 5 张截图；后续补官方博客图表裁剪 |
| P1 | 读取 Mooncake FAST 2025 论文和 checkpoint-engine 仓库 | 核验 P-D 分离、KVCache 池、早拒、热更新指标的 workload 口径 |
| P1 | 补截图资产 | 按正文截图坑位保存论文图、官方表格和 benchmark 图，补 caption |
| P2 | 设计本仓库实验表 | 把 TTFT、TPOT、cache hit、thinking token、tool calls、agent steps、失败恢复等指标写成可执行 benchmark 模板 |

### 13.4 2026-07-29 第二轮证据增强

| 时间 | 操作 | 结果 |
|---|---|---|
| 2026-07-29 12:30 CST | 下载公开 PDF 到临时目录 `/tmp/kimi_series_research_20260729` | 已抓取 K2 技术报告、K3 技术报告、Kimi k1.5 论文、Mooncake FAST 2025 论文；未写入仓库原始 PDF |
| 2026-07-29 12:31 CST | 从公开 PDF 渲染页面截图 | 已生成 9 张 PNG 到 `contexts/kimi_k2_7/Technique/assets/kimi_series/` |
| 2026-07-29 12:32 CST | 补强 K2/K3 推理效率证据 | 新增或细化 K2 MoE scaling、MuonClip/QK-Clip、K3 KDA、Gated MLA、AttnRes、Stable LatentMoE、Quantile Balancing、MoonEP、Mooncake 公式和图表 |
| 2026-07-29 12:35 CST | 补强 RL 后训练和 Agent 编排证据 | 补 K1.5 partial rollout/long2short 图表，K2 tool-use synthesis/MCP tools 图表，K3 MOPD、multi-effort RL、Agentic GRM、AgentENV 和 AET |
| 2026-07-29 12:38 CST | 增强版 Markdown 检查 | 目标文档 1622 行、131050 bytes；表格列数检查通过；代码块闭合；5 个 Mermaid；35 个技术点参考链接区；35 个待验证问题区；9 张图片引用全部存在；敏感信息检查通过 |
| 2026-07-29 12:38 CST | 增强版目标文档 hash | 目标文档 SHA256 为 `ba74ba2289fe2a7ff96c99c99601f075b9ff72be44eda8c00236405ea07031f1` |

第二轮后，P0 中“K2/K3 技术报告、Kimi Linear/KDA、K1.5 partial rollout、Mooncake FAST 论文”的关键证据已经进入正文；仍需继续做的是更细粒度截图裁剪、Kimi Linear 完整论文逐公式阅读、K2.5/PARL 原文表格摘录、Mooncake/checkpoint-engine 的 workload 口径和本仓库 benchmark 模板。

### 13.5 2026-07-29 第三轮 Kimi Linear 与 K2.5/PARL 增强

| 时间 | 操作 | 结果 |
|---|---|---|
| 2026-07-29 12:41 CST | 下载并抽取 Kimi Linear 资料 | 已读取 Kimi Linear technical report 与 GitHub README，补 KDA recurrence、混合注意力 3:1、KV cache 最高降低 75%、1M context decode 最高 6.3 倍、RULER/SpeedBench 等证据 |
| 2026-07-29 12:43 CST | 渲染 Kimi Linear 截图 | 新增 `kimi_linear_perf_speed_p1.png`、`kimi_linear_kda_algorithm_p5.png`、`kimi_linear_hybrid_ablation_p9.png`、`kimi_linear_prefill_decode_speed_p13.png` |
| 2026-07-29 12:47 CST | 下载并抽取 K2.5 technical report | 已读取 `https://arxiv.org/abs/2602.02276`，确认 30 页可抽取文本，补 joint text-vision RL、zero-vision SFT、PARL、critical steps、Agent Swarm 结果和上下文管理设置 |
| 2026-07-29 12:49 CST | 渲染 K2.5 截图 | 新增 `kimi_k25_zero_vision_rl_p4.png`、`kimi_k25_parl_agent_swarm_p5.png`、`kimi_k25_parl_critical_steps_p6.png`、`kimi_k25_dep_token_eff_rl_p10.png`、`kimi_k25_agent_swarm_results_p14.png` |
| 2026-07-29 12:53 CST | 更新目标文档 | 目标文档新增 K2.5 官方 RL objective、Toggle reward、PARL reward、critical steps、Agent Swarm Table 6、BrowseComp/WideSearch step budget、HLE/BrowseComp context management 脚注；截图引用总数增至 18 张 |
| 2026-07-29 12:56 CST | 最终校验和 141 同步 | 目标文档 1709 行、142117 bytes；代码块闭合、5 个 Mermaid、表格列数检查通过、18 张图片引用全部存在、未包含知乎账号密码；目标文档本地与 141 SHA256 均为 `0078a124ede8ec88c9b81e620e5d49f0a42de6e4fe755b9890b057a38328950e`，远端截图资产 18 张 |
