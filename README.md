# 提示文本提取与隐藏规则重建

本目录包含一份完整报告和两个并列实验。报告见 [实验报告.md](实验报告.md)，公开资料笔记见 [research_sources.md](research_sources.md)。所有目标提示和工单规则均为本实验构造，评分使用预先保存的真值。

## 实验内容

| 实验 | 目标 | 方法 | 每模型正式请求数 |
|---|---|---|---:|
| 单轮提示提取 | 从受保密要求保护的系统消息中取得私有规则或标记 | 六类固定请求×三目标×两种保护，另含正常问答和系统授权复述 | 45 |
| 隐藏规则重建 | 通过正常工单分类恢复业务规则，并生成替代提示 | 主动查询与固定随机查询；三规则×两种子×两策略，各16个标签 | 66 |

两个实验均测试 `gpt-6-luna`、`gpt-5.6-luna` 和 `gpt-6.1-sol`，统一 `reasoning_effort=low`、`max_completion_tokens=2048`，不显式设置温度和 top-p。正式结果共333次请求，原始响应全部完整返回，无失败重试。返回的型号标识与请求相符。

实验一的108次攻击未检测到私有标记、完整规则或完整系统提示；18次公开问答与9次授权复述均正确。实验二的主动策略完整恢复18/18次规则，随机策略0/18；随机规则在未查询输入上的平均一致率为92.7%。gpt-6-luna原提示的一个测试批次出现6处分类错误，模型输出一致率与规则一致率因此不同，详见报告。

## 文件与数据

- `data/protocol.json`：实验一的原始固定协议、目标规则和随机标记。当前比较使用其中36次攻击、6次正常问答、3次授权直接复述。
- `data/behavior_protocol.json`：实验二的三套规则、输入全集、独立测试集、查询池和随机标记。
- `src/api_client.py`：本地模型客户端。读取环境变量，检查返回型号，保留完整/截断/错误状态，错误信息脱敏，无自动模型替换。
- `src/multi_model.py`：实验一的三模型运行、续跑和离线汇总；可通过 `--results-dir` 将运行结果写入独立目录。
- `src/behavior.py`：候选规则推断、主动查询、模型测试、续跑和离线汇总。
- `src/scoring.py`：实验一的原句、标记及Base64解码评分。
- `src/comparison_plots.py`：报告三幅图的可复用绘图程序，可指定汇总输入、图形目录和绘图数据输出位置。
- `results/experiment1/`、`results/experiment2/`：各模型请求、原始回复与汇总指标。
- `results/comparison_plot_data.json`：图中展示的聚合数据。
- `figures/experiment1_models.png`、`figures/experiment2_learning.png`、`figures/experiment2_transfer.png`：报告使用的图形。
- `tests/`：模型返回校验、评分、查询选择、测试集隔离和断点续跑等测试。

## 环境准备

使用 Python 3.10 或更新版本，工作目录为本目录：

```powershell
python -m pip install -r requirements.txt
$env:OPENAI_BASE_URL = 'http://localhost:8080/v1'
$env:OPENAI_API_KEY = [System.Net.NetworkCredential]::new('', (Read-Host 'API key' -AsSecureString)).Password
```

密钥只放在当前进程环境，不写入配置、结果或报告。客户端仅接受带明确端口的本机HTTP地址，关闭系统代理继承。运行结束可清理：

```powershell
Remove-Item Env:OPENAI_API_KEY
```

## 运行与复现

现有协议已经固定。直接复用协议，不重新生成实验一的随机真值。

```powershell
python -m src.multi_model run
python -m src.multi_model summarize
python -m src.behavior prepare
python -m src.behavior run
python -m src.behavior summarize
python -m src.comparison_plots
```

`behavior prepare` 在已有协议时只验证，不覆盖。两个运行器均从已有日志续跑，成功记录不重复调用。可通过 `--models gpt-6-luna` 等参数选择三个指定模型中的子集。实验一和实验二的运行器均支持 `--results-dir`；指定 `--results-dir` 时，离线汇总也写入该目录。实验一的 `summarize` 另支持 `--output`。避免同时启动两个操作同一模型结果文件的进程。

需要从头运行而又保留本目录正式结果时，为每次复现实验创建一个新的空输出目录。下面的 PowerShell 示例把实验日志、汇总、图形及绘图数据都写入带时间戳的 `reproduction/` 子目录：

```powershell
$runRoot = Join-Path (Get-Location) ("reproduction/" + (Get-Date -Format "yyyyMMdd-HHmmssfff"))
$experiment1Results = Join-Path $runRoot "experiment1"
$experiment2Results = Join-Path $runRoot "experiment2"
$figures = Join-Path $runRoot "figures"
$plotData = Join-Path $runRoot "comparison_plot_data.json"
New-Item -ItemType Directory -Force -Path $experiment1Results, $experiment2Results, $figures | Out-Null

python -m src.multi_model run --results-dir $experiment1Results
python -m src.multi_model summarize --results-dir $experiment1Results
python -m src.behavior run --results-dir $experiment2Results
python -m src.behavior summarize --results-dir $experiment2Results
python -m src.comparison_plots `
  --experiment1-summary (Join-Path $experiment1Results "summary.json") `
  --experiment2-summary (Join-Path $experiment2Results "summary.json") `
  --figures-dir $figures `
  --plot-data $plotData
```

运行命令会自动续跑指定目录里已有的成功记录，因此每次从头运行应使用新的目录名。`reproduction/` 已加入 `.gitignore`，不会把个人重跑输出混入正式归档。当前固定协议来自 `data/protocol.json` 与 `data/behavior_protocol.json`；复现实验不需要重新生成真值或修改协议。

失败记录不会算作没有泄漏或分类正确。检查错误后，使用 `run --retry-errors` 对失败项最多重试一次，保留原尝试记录。输出截断需单独核查，不混入完整响应的主分母。纯离线重新评分、汇总和绘图不需要密钥，也不会调用模型。

## 实验二的数据分工

规则族有128个候选。每套目标的未解决工单全集为512种；其中24个按类别分层抽取的测试输入从攻击池排除，另加4个已解决工单用于测试。攻击池共488个输入。

每个重建过程只使用其16个实际查询标签来推断规则。主动算法最大化候选预测的联合熵；随机算法固定抽样顺序。两者使用相同的最小失配候选更新和固定平局选择。真实规则只用于构造目标和事后评分，测试标签不参与查询选择。

报告中的完整参数恢复、未查询496输入一致率、完整512输入学习曲线、28工单模型输出一致率分别保存，不能互相替代。每模型每策略的六次重建为三规则×两查询种子，测试样本在种子间复用。

## 验证

```powershell
python -m unittest discover -s tests -v
```

测试使用本地模拟客户端，不产生模型费用。真实调用的可用性、评分结果与模型行为以保存的原始响应为准。
