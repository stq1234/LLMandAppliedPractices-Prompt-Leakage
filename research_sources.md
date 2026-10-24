# 商业大模型系统提示词泄漏：公开资料与证据备忘

**用途与范围。** 本文件是公开资料研究备忘，不是课程报告，也不构成本作业的实验方案。以下区分模型在攻击下输出隐藏指令、厂商主动公开的提示词、社区整理但出处未逐条核验的文本，以及模型生成的伪似提示词。对“system prompt 泄漏”的讨论仅指隐藏配置文字或其片段被输出；这不自动证明用户数据、凭据、模型权重或完整内部上下文遭到泄漏。实际风险取决于提示词是否包含有价值且不应公开的信息；秘密凭据不应放入提示词。

## 可用案例与实证研究

### 1. Bing Chat「Sydney」：公开预览期间的直接提示注入（攻击诱导输出）

**日期与产品：** Kevin Liu 的公开记录发表于 2023-02-09，目标是 Microsoft 新版 Bing Chat（当时内部代号 Sydney）；事件报道将其提取尝试记为 2023-02-08 前后。Simon Willison 于 2023-02-15 发布同期记录，引用 Marvin von Hagen 的另一条成功会话。**提取方式：** 直接提示注入，要求模型忽略先前指令或假扮有权限的开发者，再打印此前的“document”/规则；本备忘不复述完整攻击措辞或长提示词。**可核验材料：** Willison 记录了 Sydney 代号及若干规则，并指出不同用户以不同方式获得了高度相似的输出；他也明确提出单次输出可能是模型臆造。Kevin Liu 的原始 X 帖子为 [原始记录](https://x.com/kliu128/status/1623472922374574080)，当前访问返回 403，因此本次未独立读取原帖；同期报道见 [Ars Technica，2023-02-10](https://arstechnica.com/information-technology/2023/02/ai-powered-bing-chat-spills-its-secrets-via-prompt-injection-attack/)，直接读到的同期材料见 [Simon Willison，2023-02-15](https://simonwillison.net/2023/FEB/15/bing/)。

**证据强度：中等。** 有同期用户记录与独立重复，足以支持“攻击可诱导当时产品输出与隐藏规则相符的文字”；没有 Microsoft 侧原始提示词、服务日志或逐字真实性证明，故不宜把网络流传稿当作已认证的完整部署提示词。**边界：** 这证明的是提示配置可能被诱导输出，不是用户对话或服务端秘密被窃取；后续版本和当前 Bing/Copilot 行为不在该证据范围内。

### 2. OpenAI Custom GPTs：216 个公开 GPT 的攻击评估（第一手实测）

**日期与产品：** 论文于 2023-11-20 首次提交 arXiv，2024-05-25 修订，并注明被 ICLR 2024 Secure and Trustworthy LLMs Workshop 接收；对象是 GPT Store 上线前的 OpenAI Custom GPT 功能及公开可访问的 GPT 生态。**提取方式：** 对 GPT 发起直接对抗提示，让其输出自定义指令；启用 Code Interpreter 时还测试了通过工具处理上传文件。研究者选取 16 个 OpenAI 创建的 GPT 和从第三方目录抽取的 200 个 GPT，按有无 Code Interpreter 分组，攻击最多尝试三次。**结果：** 210/216 个样本输出了系统提示文本（97.2%；无解释器 90/96，有解释器 120/120）。含文件样本中，论文报告 24/24 可提取文件（10/10 与 14/14）。原始论文：[arXiv 页面](https://arxiv.org/abs/2311.11538)、[全文](https://arxiv.org/html/2311.11538)。

**证据强度：中高。** 这是针对实际公开 GPT 的第一手实测，论文给出样本构成、重试口径、成功数和负例。**边界：** 结论限于早期 Custom GPT 生态与特定抽样和工具设置；不是对所有 Custom GPT 或基础 GPT-4 全球系统提示词的普遍估计。论文还发现一条不同的暴露路径：某些 GPT 的配置 API 曾返回文件名、大小或插件 schema；这属于产品接口元数据披露，应与“模型被诱导输出 system prompt”分开计数。论文称已预先向 OpenAI 负责任披露，写作时相关直接 API 查询已受限。

### 3. Effective Prompt Extraction：真实服务与已知真值提示词（同行评审实证研究）

**日期与产品：** Yiming Zhang、Nicholas Carlini、Daphne Ippolito 的论文发表于 COLM 2024；arXiv 首次提交于 2023-07-13，最终修订日期为 2024-08-07。论文报告对 Bing Chat、Google Bard、ChatGPT 和 Claude 3 等真实服务的黑盒提示提取测试。Claude 3 的对照真值为论文所用的 2024-03-04 公开提示词快照。**提取方式：** 在受控数据集上重复发起要求复述对话/指令的文本查询，并比较已知目标提示；面对实际服务时，作者使用要求用非英语翻译输出的策略，再把多次译文回译并比较一致性。对 Bing 的日语结果与其当时公开规则相近；Claude 3 Sonnet 的提取与公开真值相似，但少三句、多一句，作者讨论了提示版本更新这一可能性。**来源：** [COLM/OpenReview 论文](https://openreview.net/forum?id=0o95CVdNuz)、[arXiv 版本及版本日期](https://arxiv.org/abs/2307.06865)、[作者发布的数据与代码](https://github.com/y0mingzhang/prompt-extraction)。

**证据强度：高（受控实验）；中等（服务端真实提示的逐字还原）。** 论文在三类已知真值提示词与 11 个模型上评估，并提出候选提示的置信度估计；除 Alpaca-7B 外，论文报告各模型在各测试集上的提取 precision 高于 90%。对生产服务则以 Claude 3 的公开版本作比较，对其他系统更多依赖多次输出的一致性。**边界：** 论文中的商业服务测试是特定时间点、特定版本的黑盒观察，不能代表现行产品；翻译回译的相似文本不是完整的服务端取证，也不保证逐字恢复。论文自身指出，生产系统的真实提示通常未知，因而难以判定输出是否正确。

### 4. Raccoon：提示提取基准（系统化实证，不是某一产品的泄漏通报）

**日期与对象：** Findings of ACL 2024，论文刊于 2024-08；作者从多个 GPT 目录抓取逾 48,000 个条目，再均匀抽样 200 个并成功手工取得 197 个 GPT 指令提示，同时以七个模型版本（GPT-3.5、GPT-4-1106、Gemini-Pro、Llama-2-70B-Chat、Mixtral 等）评估攻击。**方法与结果：** 论文构造 14 类攻击、42 个单类攻击样例及组合攻击，覆盖无防护和加入防护模板的设置。以目标提示为真值，若 ROUGE-L recall 高于 0.8 则计为提取成功；无防护时所有模型至少有一类攻击 ASR 超过 80%，GPT-4-1106 的一个前缀注入模板报告 99%。**来源：** [ACL Anthology 书目信息](https://aclanthology.org/2024.findings-acl.791/)、[论文全文](https://aclanthology.org/2024.findings-acl.791.pdf)、[arXiv 预印本](https://arxiv.org/abs/2406.06737)。

**证据强度：高（可重复基准）；不等同于 197 个在线 GPT 被现场攻陷。** 研究把收集到的 GPT 提示作为带真值的任务提示，通过模型/API 评估提取成功率，并用少量 GPTs 检查 API 结果与自定义 GPT 行为相似。它支持攻击类型、评价口径和模型间差异的比较；不支持把某个汇总 ASR 解释为某个现存商业服务的攻击率。ROUGE-L 的阈值也表示近似匹配，不等价于逐字完整泄漏。

## 必须分开的其他文本类别

### 厂商主动公开：不是攻击泄漏

Anthropic 官方在 Claude Platform Docs 中主动发布了 claude.ai 与 iOS/Android 应用使用的 dated system prompt。以 Claude Sonnet 3.5 页面为例，官方留有 2024-10-22 和 2024-11-22 两个版本；总览明确说明这些提示会定期更新，且这些更新**不适用于 Claude API**。因此它是产品方公开披露特定消费端版本提示词的实例，不能标作“被攻击泄漏”，也不能把该文本外推到 API、所有客户端隐藏消息或其他模型版本。来源：[官方提示词总览](https://platform.claude.com/docs/en/release-notes/system-prompts/overview)、[Sonnet 3.5 dated prompt](https://platform.claude.com/docs/en/release-notes/system-prompts/claude-sonnet-3-5)。Anthropic 对 model system cards 的说明则是记录能力、安全评估与部署决策，并不意味着每一份 card 都含逐字运行提示词：[官方 system cards 目录](https://www.anthropic.com/system-cards)。

OpenAI 的 [Model Spec](https://github.com/openai/model_spec/blob/main/model_spec.md) 与 [Instruction Hierarchy 论文](https://openai.com/index/the-instruction-hierarchy/)也是厂商主动公开的规范/方法资料，可为权限层级和行为目标提供依据；它们不是某个线上请求的完整逐字隐藏上下文。该论文发布日期为 2024-04-19，提出当不同权限消息冲突时优先执行高权限消息，并在 GPT-3.5 上训练验证。

### 社区整理、来源未逐条核验的摘录

例如 [YeeKal/leaked-system-prompts](https://github.com/YeeKal/leaked-system-prompts) 在本次访问（2026-10-24）时列出 33 家公司的 116 条提示，并把社区档案、Anthropic 官方提示词等列为交叉来源。它是寻找线索的资料索引，不是每一条提示都附有可独立复现的攻击记录、采集时间、目标版本、服务器真值或厂商确认。条目名里的“leaked”、与正式提示词相似、其他仓库重复转载，都不能单独证明它是攻击泄漏；如来源属于厂商正式公开，应按“厂商公开”标注；否则记为社区声称/未核验摘录。

### 模型臆造或近似重构

仅凭模型回答“这是我的 system prompt”或输出格式完整的长规则，不足以证明文本来自真实隐藏上下文。Bing 同期案例中 Simon Willison 明确保留臆造的可能；Effective Prompt Extraction 则把“是否为真提示”作为独立评价问题，用已知真值验证预测器，并对没有真值的实际服务谨慎使用一致性作为支持证据。研究记录应分别标注：逐字匹配已知真值、与可验证公开版本近似、多个查询间重复出现、以及只有模型自称真实。后两类不能写成已认证的“完整提示泄漏”。

## GitHub 提示集合的实读样本与文本对比（2026-10-24）

### 证据口径与阅读范围

本次逐页阅读了三个社区整理仓库中的具体文件：[`asgeirtj/system_prompts_leaks`](https://github.com/asgeirtj/system_prompts_leaks)、[`elder-plinius/CL4R1T4S`](https://github.com/elder-plinius/CL4R1T4S) 和 [`x1xhlol/system-prompts-and-models-of-ai-tools`](https://github.com/x1xhlol/system-prompts-and-models-of-ai-tools)，并核对了 xAI 与 Anthropic 的官方公开材料。下列社区文件均能证明相应文字确实出现在被阅读的仓库版本中；仅凭仓库自称“leaked”、文件细节丰富或多个仓库互相转载，不能证明它来自该产品的真实部署上下文。社区文件若无可核验采集记录、厂商确认或已知提示真值，证据只支持“社区归档文本如此”，不支持“产品真实提示词已被完整提取”。

官方来源则可证明厂商公开了特定文字及其适用范围，但这种主动公开不属于攻击泄漏。以下版本范围均按官方页面或文件所标明的产品、模型和日期限定；社区文件未标版本日期时，明确记录为读取的 `main` 分支快照，不推定它就是当前线上版本。

### 五个具体文本及其可观察特征

| 产品与文件上下文 | 直接阅读到的规则特征 | 证据等级与边界 |
|---|---|---|
| Anthropic Claude Sonnet 3.5，官方 [2024-11-22 dated system prompt](https://platform.claude.com/docs/en/release-notes/system-prompts/claude-sonnet-3-5) | 开头标定身份、知识更新时间和当前日期变量；说明该版本不能直接打开 URL；要求在引用资料时提示可能产生错误引用；对复杂问题提供逐步处理、对简单任务简洁作答，并规定少问追问、Markdown 和风险请求的回应方式。 | **厂商官方公开。** 官方总览说明这是 claude.ai 与移动应用的核心提示更新，定期变更且不适用于 Claude API；该页面存在多个日期版本，不能代表当前 Claude 版本或 API 隐藏上下文。[官方适用范围](https://platform.claude.com/docs/en/release-notes/system-prompts/overview) |
| xAI Grok 4 chat，官方 [`grok4_system_turn_prompt_v8.j2`](https://github.com/xai-org/grok-prompts/blob/main/grok4_system_turn_prompt_v8.j2) | 文件含 Jinja 条件分支（如移动端、主观问题）；将高优先级安全规则置于常规角色指令之前；列出可分析的 X 资料及用户上传文件，要求生成图像前确认；含订阅/API 产品问题的转介规则、X 搜索要求以及按条件使用表格等输出约束。 | **厂商官方公开。** [仓库 README](https://github.com/xai-org/grok-prompts) 将该文件明确列为 grok.com 和 X 上 Grok 4 助手的 system prompt；这只能支持该仓库发布的模板内容，不证明每次线上请求的所有动态上下文。相邻的 [`grok_4_safety_prompt.txt`](https://github.com/xai-org/grok-prompts/blob/main/grok_4_safety_prompt.txt) 则被 README 标为 `grok-4-0709` API 模型的安全提示前缀，表明同一品牌在产品聊天与 API 组件之间有不同文本，不能合并成一份通用提示。 |
| Perplexity Search，社区文件 [`Perplexity/perplexity-search.md`](https://github.com/asgeirtj/system_prompts_leaks/blob/main/Perplexity/perplexity-search.md)；仓库更新表将该条目标为 2026-10-23 | 文本不只规定“助手人设”，还要求先把请求路由到直接回答或工具调用；事实、建议、实时状态等问题优先调用工具；根据工具结果逐点引用；连接器访问、外部数据查询及写入类动作另有发现工具、确认和授权流程。 | **社区声称、未独立认证。** 直接阅读可确认仓库文件含这些业务流程，但无法据此确认 Perplexity 当前线上配置、每个工具定义或安全执行均与此一致。文件有显式角色区、工具流程、引用规则和外部操作步骤，是判断其产品行为编排颗粒度的实例。 |
| Cursor Chat，社区文件 [`Cursor Prompts/Chat Prompt.txt`](https://github.com/x1xhlol/system-prompts-and-models-of-ai-tools/blob/main/Cursor%20Prompts/Chat%20Prompt.txt)；正文自称基于 GPT-4o，未标注采集日期 | 角色是 IDE 编程助手；提示称用户消息可能附带打开文件、光标位置、最近查看内容、编辑历史和 lint 错误；请求由结构标签承载，并有工具调用 schema、代码编辑输出格式及工具使用限制。 | **社区声称、未独立认证。** 仓库文件是 119 行的 Chat Prompt，不含可核验的部署版本标识；它展示了“系统指令 + 动态 IDE 状态 + 工具协议”如何共同构成产品控制面，但不能证明所列运行时细节对应当前 Cursor 版本。 |
| Cluely，社区文件 [`CLUELY/Cluely.mkd`](https://github.com/elder-plinius/CL4R1T4S/blob/main/CLUELY/Cluely.mkd)；`main` 分支，文件无版本日期 | 角色是分析屏幕内容的助手；对模糊意图要求输出固定开场和有限猜测，避免主动总结屏幕；按编程、数学、邮件和界面导航分流，并规定代码注释、答案格式和操作步骤的粒度。 | **社区声称、且存在来源污染疑点。** 文件末尾在“用户提供上下文”段中混入一条要求模型输出系统提示的攻击性指令。它可能是提示样本中的注入上下文、社区采集时附入的内容或未经净化的重构片段，单凭此文件无法确定；不应把这一段当作 Cluely 厂商指令或成功泄漏的证据。 |

### 横向观察

这五份文本显示，产品提示往往承担工作流编排，而非只描述人格。Perplexity 文件重点写“何时查资料、何时调用连接器、如何引用和确认外部写入”；Cursor 把 IDE 状态和代码编辑协议纳入输入输出约定；Cluely 则围绕屏幕理解和任务类别规定何时回答及如何格式化；Grok 在同一模板中用条件分支连接安全规则、产品工具和不同客户端要求。由此可见，提示暴露的商业价值可能在工具边界、信息流和产品流程中，而不只在角色文案。

安全表达也并非一种统一模板。所读 Grok 4 文本显式设置高优先级安全段，并描述对提示注入、编码或角色扮演式规避的处理；Anthropic 2024 dated 版本强调风险话题可在教育、事实或创作语境下提供帮助，并以明确伤害意图作为重要判断；Cluely 的社区文件将模糊屏幕意图设为单独响应状态。它们体现的是不同任务和版本下的规则取舍，不能据此对各厂商整体安全表现或当前产品效果作排名。

这批样本还显示了“完整提示”主张的验证难点：模板变量和 Jinja 条件需要知道运行上下文才能恢复实际渲染结果；社区文件可能把角色规则、工具 schema、运行时说明和注入样例拼在一起；一个详细、连贯的文本也可能是重构或伪造。因此报告宜把“直接观察到的社区文件内容”与“已知真值上的攻击提取结果”分开陈述。若要评价泄漏，至少应说明目标版本、来源链、真值或人工 canary、命中标准和重复查询口径；仅有长文本输出或仓库副本不能确认逐字真实性。

## 提取、功能逆向与间接注入的研究边界

以下术语对应不同攻击目标，不宜将论文中的成功率合并为一个“提示泄漏率”。

| 研究 | 攻击目标与方法 | 与提示文本核读的关系 |
|---|---|---|
| [Effective Prompt Extraction（COLM 2024）](https://openreview.net/forum?id=0o95CVdNuz) 与 [Raccoon（Findings of ACL 2024）](https://aclanthology.org/2024.findings-acl.791/) | 在可获得真值的提示集上设计/组合提取请求，并以提示匹配指标评估；重点是文本恢复与攻击策略比较。其结果不能直接解释为所有生产应用的攻击概率。 | 可支持“直接诱导或迭代查询能否恢复隐藏文本”的受控证据；本笔记前文已有数据、口径与范围说明。 |
| [PLeak（CCS 2024）](https://arxiv.org/html/2405.06823) | 自动化闭盒提示泄漏：先用影子提示优化对抗查询，再向目标应用发送变换后的查询，并聚合/后处理多个回复重建目标提示。论文报告其多查询设置在所测 Poe 应用中 exact-match 重建率为 68%。 | 属于直接查询并以已知或目标提示作重建的攻击，与纯手工“请复述规则”不同；结果适用于论文所测时期的 Poe 应用和模型，不可外推到现行商业产品。 |
| [RepeatLeakage（AAAI 2025）](https://ojs.aaai.org/index.php/AAAI/article/view/34832) | 利用模型执行重复任务时依据上下文复现文本的现象，通过限定复述范围并要求保持格式，尝试输出系统提示及对话上下文。作者报告在公开 GPT Store/ChatGPT 场景测试。 | 是一种文本泄漏方法；仍需用外部真值或特定上下文证据验证输出，模型自称真实不构成确认。 |
| [PRSA（USENIX Security 2025）](https://www.usenix.org/system/files/usenixsecurity25-yang-yong.pdf) | **提示窃取/功能逆向**：只从有限输入—输出样本推断提示的意图，再生成并剪枝一个可复现输出风格、主题和结构的替代提示。论文将输入输出一致性作为核心目标；这并不要求恢复原提示原文。 | 回答的是“能否复制某服务功能”，不是“是否逐字泄漏了系统提示”。作者仓库说明真实购买提示和被测 GPT 未公开提供，演示使用合成数据以降低法律和伦理风险：[PRSA 实验代码说明](https://github.com/yangyZJU/PRSA)。 |
| [AgentDojo（NeurIPS 2024）](https://proceedings.neurips.cc/paper_files/paper/2024/hash/97091a5177d8dc64b1da8bf3e1f6fb54-Abstract-Datasets_and_Benchmarks_Track.html) | 研究恶意指令嵌入外部工具返回数据后，能否劫持使用工具的 agent；基准包含 97 项现实任务和 629 个安全测试案例。主要安全目标是约束 agent 对邮件、银行网站等不可信内容的操作。 | 属于**间接提示注入与工具安全**评测；它可以导致错误工具调用或隐私/数据风险，但不以系统提示原文恢复为必要成功条件，因此不等价于提示抽取。 |

据此，常见方法至少分成四类：手工直接注入/越权复述请求；通过多轮、格式/翻译变换和回复聚合减少拒答影响；学习或优化对抗查询（PLeak）；以及只观察输入输出、构造功能等价替代提示（PRSA）。若攻击指令藏在网页、邮件或其他工具结果中并诱导 agent 行动，应单列为间接注入（AgentDojo）。这些方法的攻击目标和成功判据不同。课程实验若只测试 canary 是否出现在直接对话输出中，结论应限制在该文本泄漏面，不应声称覆盖商业提示克隆、工具执行或所有注入场景。

### 可直接用于报告的研究发现

1. 商业提示文本承担的不只是人设描述，还包含搜索路由、工具调用、外部操作确认、动态客户端上下文和输出规范；Perplexity、Cursor 与 Grok 的实读文件提供了具体例证。[Perplexity 文件](https://github.com/asgeirtj/system_prompts_leaks/blob/main/Perplexity/perplexity-search.md)、[Cursor 文件](https://github.com/x1xhlol/system-prompts-and-models-of-ai-tools/blob/main/Cursor%20Prompts/Chat%20Prompt.txt)、[xAI 官方 Grok 4 文件](https://github.com/xai-org/grok-prompts/blob/main/grok4_system_turn_prompt_v8.j2)
2. 系统提示的目标产品和版本边界十分关键：Anthropic 明确区分 Claude.ai/移动应用与 API；xAI 将 Grok 4 聊天提示与单独的 `grok-4-0709` API 安全前缀分列。[Anthropic 总览](https://platform.claude.com/docs/en/release-notes/system-prompts/overview)、[xAI README](https://github.com/xai-org/grok-prompts)
3. 社区仓库适合用来分析公开流传文本的设计特征，但无法单独证明厂商部署真实性；Cluely 社区样本还包含嵌在用户上下文区的提示抽取指令，说明采集/转存污染会让来源判断更加困难。[Cluely 文件](https://github.com/elder-plinius/CL4R1T4S/blob/main/CLUELY/Cluely.mkd)
4. 直接泄漏与功能克隆要分开：PLeak 的 exact-match 复原衡量文本恢复；PRSA 的输入输出一致性衡量能否复制提示功能，不要求找回原文。[PLeak](https://arxiv.org/html/2405.06823)、[PRSA](https://www.usenix.org/system/files/usenixsecurity25-yang-yong.pdf)
5. 输出中出现一段结构完整的提示文本并不足以证明真实性；能获得目标真值时应报告精确/规范化匹配或近似匹配，否则应把结果表述为模型复述或与公开版本相似，而不是认证的完整泄漏。[Effective Prompt Extraction](https://openreview.net/forum?id=0o95CVdNuz)、[Raccoon](https://aclanthology.org/2024.findings-acl.791/)
6. 对工具型 agent 的间接注入应单列评估，因为恶意工具数据可能劫持行动，即使最终没有输出系统提示。[AgentDojo](https://proceedings.neurips.cc/paper_files/paper/2024/hash/97091a5177d8dc64b1da8bf3e1f6fb54-Abstract-Datasets_and_Benchmarks_Track.html)

## 文本恢复的评价口径

设隐藏提示为 \(P\)，接口累计输出为 \(R\)。只有将输出与外部已知目标文本对照，才能判定是否恢复了原提示；模型自称其回答是真实提示不足以证明来源。Raccoon 将 ROUGE-L recall \(>0.8\) 作为该基准的提取成功阈值，这是论文的操作化定义，不是通用安全标准。逐字/规范化匹配、近似文本重合和功能一致性衡量不同目标，应分别表述。

## 主要一手来源索引

- Simon Willison，Bing Chat 事件同期记录，2023-02-15：[页面](https://simonwillison.net/2023/FEB/15/bing/)。
- Yu et al.，Custom GPTs 216 样本实测，arXiv:2311.11538（2023-11-20；v2 2024-05-25；ICLR 2024 workshop）：[全文](https://arxiv.org/html/2311.11538)。
- Zhang et al.，Effective Prompt Extraction，COLM 2024：[OpenReview](https://openreview.net/forum?id=0o95CVdNuz)、[arXiv](https://arxiv.org/abs/2307.06865)。
- Wang et al.，Raccoon，Findings of ACL 2024：[ACL Anthology](https://aclanthology.org/2024.findings-acl.791/)、[PDF](https://aclanthology.org/2024.findings-acl.791.pdf)。
- Anthropic，Claude.ai/mobile 公开提示词版本及适用范围：[官方总览](https://platform.claude.com/docs/en/release-notes/system-prompts/overview)、[Sonnet 3.5 版本页](https://platform.claude.com/docs/en/release-notes/system-prompts/claude-sonnet-3-5)。
- OpenAI，公开行为规范与指令层级研究：[论文/说明](https://openai.com/index/the-instruction-hierarchy/)、[Model Spec](https://github.com/openai/model_spec/blob/main/model_spec.md)。
