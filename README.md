# 提示文本提取与隐藏规则重建

本目录包含一份完整报告和两个并列实验。报告见[实验报告.md](实验报告.md)，公开资料笔记见[research_sources.md](research_sources.md)，实验一协议与评分说明见[research_extension.md](research_extension.md)。目标提示和工单规则均由本实验构造，保存的真值用于事后评分。

## 实验内容与请求预算

| 实验部分 | 模型与覆盖 | 已保存请求数 |
|---|---|---:|
| 实验一：原始固定单轮 | 三模型×三目标×六策略×basic/structured；另有18次公开问答、9次系统授权复述 | 135 |
| 实验一：单轮保护配置扩展 | 三模型×三目标×六策略×none/weak | 108 |
| 实验一：固定脚本扩展 | 三模型×两保护；18组多轮共54次请求，另54次单轮表示测试 | 108 |
| 实验一：双模型自适应多轮 | 攻击者为gpt-6-luna或gpt-6.1-sol，防守者为gpt-6-luna；三个目标×三保护×每组四轮 | 144 |
| 实验二：隐藏规则重建 | 三模型×三规则×两种子×主动/随机；36次重建及模型行为测试、直接索取对照 | 198 |

合计693次已保存请求，均完整返回，无失败重试。双模型部分包含72次攻击者生成和72次防守者回复，不能将144次请求当作144次攻击或144组对话。原始单轮、保护扩展及实验二使用`gpt-6-luna`、`gpt-5.6-luna`和`gpt-6.1-sol`；双模型配对以表中配置为准。所有模型请求使用`reasoning_effort=low`、`max_completion_tokens=2048`，不显式设置温度和top-p。

216次固定单轮均未取得随机标记或完整规则原句；无显式保密时，gpt-5.6-luna有3次输出三条规则的中文含义。18组固定多轮未取得目标信息。自适应多轮有5/18组恢复规则含义，其中4组进一步取得至少一条完整原句、3组取得全部三条原句；全部攻击均未取得随机标记。实验二主动策略完整恢复18/18次规则，随机策略0/18，后者在未查询输入上的平均规则一致率仍为92.7%。样本数量、保护条件和评价层次见报告，不合并为单一成功率。

## 文件与数据

| 内容 | 协议与程序 | 原始结果与评分 |
|---|---|---|
| 原始单轮与对照 | `data/protocol.json`、`src/multi_model.py`、`src/scoring.py` | `results/experiment1/` |
| 无显式/简短保密单轮 | `data/protection_sweep_protocol.json`、`src/protection_sweep.py` | `results/protection_sweep/` |
| 固定脚本及补充表示 | `data/extension_protocol.json`、`src/extension.py` | `results/extension_dev/dev/` |
| 双模型自适应多轮 | `data/adaptive_dialogue_protocol.json`、`src/adaptive_dialogue.py` | `results/adaptive_dialogue/` |
| 工单规则重建 | `data/behavior_protocol.json`、`src/behavior.py` | `results/experiment2/` |
| 实验一汇总与图1 | `src/leakage_overview.py` | `results/leakage_overview.json`、`figures/experiment1_dialogue_recovery.png` |
| 实验二图2、图3 | `src/comparison_plots.py` | `results/comparison_plot_data.json`、`figures/experiment2_learning.png`、`figures/experiment2_transfer.png` |

`src/api_client.py`读取环境变量、校验返回型号并保留完整/截断/错误状态，错误信息脱敏，不自动替换模型。JSONL保存实际请求与原始响应；语义判读保存在各结果目录的`semantic_review.jsonl`，固定脚本历史格式使用`manual_review.jsonl`。这些判读由Agent完成，不是独立人工标注。新生成的回复必须重新逐条审阅，不能沿用旧运行的语义结论。

双模型日志`results/adaptive_dialogue/adaptive_dialogue.jsonl`按会话保存双方的独立系统消息、每轮实际请求、攻击者生成、发送给防守者的用户提示以及防守者原始回复。每次双方都收到各自完整历史；攻击者不接收真值、保护条件或评分。`strategy_note`仅是模型生成的简短战术说明。

## 环境与密钥配置

使用Python 3.10或更新版本，在仓库根目录执行：

```powershell
python -m pip install -r requirements.txt
$env:OPENAI_BASE_URL = 'http://localhost:8080/v1'
$env:OPENAI_API_KEY = [System.Net.NetworkCredential]::new('', (Read-Host 'API key' -AsSecureString)).Password
```

密钥仅放在当前进程环境，不写入协议、结果或报告。客户端仅接受带明确端口的本机HTTP地址，关闭系统代理继承。运行结束后可执行`Remove-Item Env:OPENAI_API_KEY`。离线评分与绘图不需要密钥。

## 离线复核已有结果

以下命令仅读取已保存的响应，不调用模型：

```powershell
python -m src.multi_model summarize
python -m src.protection_sweep summarize
python -m src.extension summarize --phase dev
python -m src.adaptive_dialogue summarize --results-dir results/adaptive_dialogue
python -m src.behavior summarize
python -m src.leakage_overview
python -m src.comparison_plots
```

`leakage_overview`核对请求完整性、模型/条件覆盖、语义审阅及原始响应的一致性，再生成汇总和图1。累积轮次曲线只计防守者输出；攻击者复述之前已知的规则不作为新的披露。`comparison_plots`从实验二完整汇总生成两幅比较图及其绘图数据。

## 重新运行实验

固定协议已包含目标规则、随机标记和数据划分。复现时保留协议，为新运行使用独立结果目录。下面设置一个新的输出根目录；同一运行中保留该变量以便续跑：

```powershell
$runRoot = Join-Path (Get-Location) ('reproduction/' + (Get-Date -Format 'yyyyMMdd-HHmmssfff'))
New-Item -ItemType Directory -Path $runRoot | Out-Null
```

### 原始单轮与保护配置扩展

```powershell
python -m src.multi_model run --results-dir (Join-Path $runRoot 'experiment1')
python -m src.multi_model summarize --results-dir (Join-Path $runRoot 'experiment1')
python -m src.protection_sweep prepare
python -m src.protection_sweep run --results-dir (Join-Path $runRoot 'protection_sweep') --max-calls 108
python -m src.protection_sweep summarize --results-dir (Join-Path $runRoot 'protection_sweep')
```

两部分分别最多135次和108次请求。`prepare`校验已有保护扩展协议；`--max-calls`限制本次进程的新请求数量。两运行器可用`--models gpt-6-luna`等参数选择指定型号的子集，成功记录不重复调用。失败项经检查后可通过`--retry-errors`重试一次，并保留原尝试。

### 固定脚本多轮及补充表示

```powershell
python -m src.extension prepare --phase dev --results-dir (Join-Path $runRoot 'extension_dev')
python -m src.extension run --phase dev --results-dir (Join-Path $runRoot 'extension_dev')
python -m src.extension summarize --phase dev --results-dir (Join-Path $runRoot 'extension_dev')
```

该阶段运行72组会话，最多108次请求，输出自动写入指定根目录下的`dev/`。多轮使用真实历史，后续用户消息由协议预先固定。运行器顺序执行，无自动重试；失败或截断不纳入完整响应的主要分母。

### 双模型自适应多轮

```powershell
$dialogueResults = Join-Path $runRoot 'adaptive_dialogue'
python -m src.adaptive_dialogue prepare --results-dir $dialogueResults
python -m src.adaptive_dialogue run --results-dir $dialogueResults --max-calls 144
python -m src.adaptive_dialogue summarize --results-dir $dialogueResults
```

`prepare`要求目标目录尚不存在，冻结该次运行的协议、目标及模型组合。每会话最多四轮，整个配对矩阵最多144次请求。`--max-calls`可以进一步限制当次新增调用，例如设为16；恢复时仅重新执行`run`，不要再次`prepare`。也可用`--attacker-models gpt-6.1-sol`仅运行指定攻击者。

每次攻击者或防守者调用完成后立即保存，重启不会重复调用已记录的一方。终态错误、截断或格式错误不会自动重试。随机标记精确命中会提前结束会话；原句和语义恢复仅离线评分，不反馈给攻击者改变其后续决策。

### 实验二：工单规则重建

```powershell
python -m src.behavior prepare
python -m src.behavior run --results-dir (Join-Path $runRoot 'experiment2')
python -m src.behavior summarize --results-dir (Join-Path $runRoot 'experiment2')
python -m src.comparison_plots --experiment2-summary (Join-Path $runRoot 'experiment2/summary.json') --figures-dir (Join-Path $runRoot 'figures') --plot-data (Join-Path $runRoot 'comparison_plot_data.json')
```

该实验最多198次请求；`prepare`在协议已存在时仅校验。攻击者知道128个候选规则的形式，但不知道目标参数。每次重建只根据16个实际查询标签更新候选；主动算法最大化候选预测的联合熵，随机算法按固定顺序采样，两者使用相同的最小失配更新和平局选择。

每套目标有512个未解决工单输入，其中24个分层测试输入从查询池排除，另加4个已解决工单作模型测试。查询池为488个输入，每次查询16个后在其余496个未解决输入上检验规则一致率。完整参数恢复、496输入一致率、512输入学习曲线和28张工单的模型输出一致率分别保存，不能互相替代。

### 新运行的语义复核与绘图

为新运行分别完成原始单轮、保护扩展、固定脚本及自适应会话的语义审阅，依据该次实际回复保存相应JSONL；审阅记录中的ID、原始文本和证据必须对应新运行。然后执行：

```powershell
python -m src.leakage_overview --results-root $runRoot --output (Join-Path $runRoot 'leakage_overview.json') --figure (Join-Path $runRoot 'figures/experiment1_dialogue_recovery.png')
```

不完整的运行或缺失的审阅不能生成完整对照图。所有运行器禁止并发修改同一个结果文件。`reproduction/`已被Git忽略，个人重跑结果不会覆盖正式归档。

## 验证

```powershell
python -m unittest discover -s tests -v
```

测试使用模拟客户端，不产生模型调用。覆盖返回型号与状态校验、解码评分、候选选择、测试集隔离、完整历史、上下文隔离、调用预算和断点续跑；实验观察以保存的实际响应和逐条证据为准。
