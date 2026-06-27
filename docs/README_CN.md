<div align="center">

<img src="../sources/alphaevo_terminal_v1_agent_evolution.svg" alt="AlphaEvo Logo" width="680">

# 🧬 AlphaEvo

**一个开源的股票策略自我进化研究 Agent**

*让策略经历诊断、改写、复测和剪枝，而不是只生成一份 YAML*

[![GitHub stars](https://img.shields.io/github/stars/ZhuLinsen/alphaevo?style=social)](https://github.com/ZhuLinsen/alphaevo/stargazers)
[![CI](https://github.com/ZhuLinsen/alphaevo/actions/workflows/ci.yml/badge.svg)](https://github.com/ZhuLinsen/alphaevo/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](../LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](../Dockerfile)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](../CONTRIBUTING.md)

[**项目概览**](#-项目概览) · [**核心能力**](#-核心能力) · [**快速开始**](#-快速开始) · [**架构**](#-系统架构) · [**真实验证**](#-真实验证结果2026-年-4-月-10-日) · [**CLI 命令**](#-cli-命令一览)

[English](../README.md) | [中文](README_CN.md)

</div>

---

## ✨ 项目概览

AlphaEvo 用历史市场数据回测股票策略，并把策略放进完整研究闭环：回测、失败归因、结构化改写、再次验证、保留证据链。它把可执行 YAML DSL、防过拟合评估、研究日志和进化树整合到同一条研究工作流里。

### 可复现无 API Showcase

内置 frozen yfinance 快照，美股科技股篮子，2025-02-11 到 2026-04-10：

| 策略 | 信号数 | 胜率 | 平均收益 | 最大回撤 | 评分 |
|------|--------|------|----------|----------|------|
| `rsi_reversion_v1` baseline | 0 | 0.0% | 0.00% | 0.0% | 8.1% |
| `rsi_reversion_v7` champion | 37 | 48.6% | 2.94% | 23.6% | 68.7% |

过程：研究委员会先判断入场条件过度确认，然后依次解锁信号、放宽止损、延长持仓、提高止盈倍数、把支撑锚点切到 MA60、收紧放量确认。每次修改都必须复测后才接受。该 champion 在快照上的 train-val / val-test gap 为 7.1% / 7.1%。完整报告见：[showcase_rsi_reversion_real_snapshot.md](reports/showcase_rsi_reversion_real_snapshot.md)。

### 实时数据 LLM 进化路径

实时 yfinance 数据，配置外部 LLM provider 后运行 `alphaevo evolve rsi_reversion_v1 --method llm`，时间为 2026-04-10：

| 策略 | 信号数 | 胜率 | 平均收益 | 评分 |
|------|--------|------|----------|------|
| `rsi_reversion_v1` baseline | 0 | 0.0% | 0.00% | 8.1% |
| `rsi_reversion_v3` LLM champion | 498 | 52.6% | 1.22% | 56.3% |

不同数据窗口和协议不应该直接混排；快照 showcase 是稳定的无 API 首跑，LLM 路径展示的是实时市场数据上的研究闭环。

## 🧠 核心能力

- **回测与评估**：在真实数据上运行策略，完成采样、多维评分和防过拟合检验。
- **LLM 引导的策略进化**：诊断失败模式，提出定向修改，并重新验证新版本是否确实更优。
- **确定性研究委员会**：技术面、风险、过拟合、数据质量、mutation planner 输出结构化结论，无需 API Key。
- **可追踪的研究流程**：为每轮迭代保留报告、LLM 证据、进化树和 trajectory 导出。

## 🚀 快速开始

### 从源码开始，30 秒体验（无需 API Key！）

```bash
git clone https://github.com/ZhuLinsen/alphaevo.git
cd alphaevo
pip install -e .
alphaevo showcase
```

<p align="center">
  <img src="demo.gif" alt="AlphaEvo Demo" width="720">
</p>

观看真实历史数据上的策略自我进化：

```
Showcase Chain: baseline + up to 6 validated mutations

Round 1 │ rsi_reversion_v1 │ 信号: 0  │ 评分: 8.1%
  研究委员会：入场条件过严
  修改：entry.logic and -> or

Round 2 │ rsi_reversion_v2 │ 信号: 79 │ 评分: 17.8%
  修改：exit.stop_loss.value 0.05 -> 0.08

Round 7 │ rsi_reversion_v7 │ 信号: 37 │ 评分: 68.7% 🏆
```

### 选择合适的入口

| 目标 | 命令 | 数据 | 是否需要 LLM |
|------|------|------|---------------|
| 真实数据 Showcase | `alphaevo showcase` | 内置 yfinance 快照 | 否 |
| 30 秒体验 | `alphaevo demo` | 内置 yfinance 快照 | 否 |
| 合成数据开发烟测 | `alphaevo demo --synthetic` | 合成数据 | 否 |
| 一句话策略转可执行 YAML | `alphaevo strategy draft "<idea>" --save` | 无 | 否 |
| 一句话策略直接起草、回测、优化 | `alphaevo strategy research "<idea>"` | 真实数据 | 否 |
| 按想法修订已有策略并验证 | `alphaevo strategy improve <id> "<改法>"` | 真实数据 | 否 |
| 实时市场数据冒烟验证 | `alphaevo showcase --live` 或 `alphaevo demo --real` | yfinance / akshare | 否 |
| 更完整的真实数据回测 | `alphaevo run ma_crossover_v1` | yfinance | 否 |
| 测试突破/波动收缩模板 | `alphaevo run volatility_compression_breakout_v1` | yfinance | 否 |
| 优化买点阈值和卖点/风控规则 | `alphaevo optimize <id> --spaces entry,params,indicator,exit,stoploss,takeprofit,holding` | 真实数据 | 否 |
| 平衡胜率和盈利质量 | `alphaevo optimize <id> --objective quality --min-win-rate 0.5 --min-avg-return 0 --min-profit-loss-ratio 1.0 --max-drawdown 0.35 --min-signals 30 --param-max-changes 2 --max-values-per-param 8 --evaluation-mode fast --full-eval-top 5` | 真实数据 | 否 |
| 提高收益质量 | `alphaevo optimize <id> --objective profit_quality --min-win-rate 0.5 --min-avg-return 0.006 --min-total-return 0.18 --min-profit-loss-ratio 1.1 --joint-top 4 --parallel-workers 4` | 真实数据 | 否 |
| 过滤疑似过拟合候选 | `alphaevo optimize <id> --objective robust_profit_quality --evaluation-mode fast --full-eval-top 8 --reject-overfit --max-train-val-gap 0.12 --max-val-test-gap 0.10 --max-walk-forward-gap 0.12 --min-walk-forward-pass-rate 0.5` | 真实数据 | 否 |
| 联合优化买点和卖点 | `alphaevo optimize <id> --spaces all --objective quality --joint-top 3 --joint-candidates-per-seed 64` | 真实数据 | 否 |
| 旗舰研究智能体路径 | `alphaevo evolve <id> --method llm --output reports/` | 真实数据 | 是 |

真实数据命令需要额外安装数据适配器：默认美股流程装 `pip install -e ".[data-yfinance]"`，A 股装 `pip install -e ".[data-akshare]"`，两者都要就装 `pip install -e ".[data-full]"`。

`alphaevo showcase` 是稳定的真实数据首跑：使用内置 yfinance 快照并生成可分享报告。`alphaevo showcase --live` 会先尝试实时 yfinance，失败则回退到快照。想先看一条完整回测的话跑 `alphaevo run ma_crossover_v1`。

`--objective robust_profit_quality` 会在收益质量上叠加稳定性评分。`--reject-overfit`、`--max-train-val-gap`、`--max-walk-forward-gap` 等稳健门槛依赖完整候选指标。使用 fast 搜索时，需要把 `--full-eval-top` 设到足够覆盖你想判断的头部候选。

### 当前上线范围

- **最成熟的路径**：`rsi_reversion_v1` 和 `ma_crossover_v1` 用真实数据跑进化
- **最稳的策略类型**：trend + reversal，核心信号来自 OHLCV 和 benchmark
- **偏收益弹性的趋势模板**：`volatility_compression_breakout_v1` 使用前高突破触发、区间位置过滤、波动收缩过滤和 trailing 止盈
- **实验性的**：event + rotation 策略依赖 proxy 做新闻/板块语义，当研究预览看就好

### 完整安装（含 LLM 进化功能）

```bash
# 在仓库根目录补齐 LLM + 默认真实数据能力
git clone https://github.com/ZhuLinsen/alphaevo.git
cd alphaevo
pip install -e ".[llm,data-yfinance]"

# 设置 LLM API Key（进化功能需要）
export ALPHAEVO_API_KEY=your_api_key
export ALPHAEVO_LLM_MODEL=gemini/gemini-2.0-flash  # 或 openai/gpt-4o 等

# 在默认 yfinance 适配器上运行一条真实研究闭环
alphaevo run ma_crossover_v1

# 无需 API Key 的真实数据 demo
alphaevo demo --real

# 跑旗舰 LLM 研究路径
alphaevo evolve rsi_reversion_v1 --method llm --rounds 3 --output reports/rsi_evolve/

# 查看策略排行榜
alphaevo leaderboard
```

如果你还想跑内置 A 股策略，可以额外执行 `pip install -e ".[data-akshare]"`，或者直接安装 `pip install -e ".[llm,data-full]"`，然后使用 `alphaevo run trend_pullback_rebound_v1 --adapter akshare`。

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                    CLI (Typer + Rich)                     │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│              编排层 Orchestrator (Pipeline)               │
│   生成 → 采样 → 回测 → 评估 → 反思 → 进化 → 排行榜     │
└──┬────────┬────────┬────────┬────────┬────────┬─────────┘
   │        │        │        │        │        │
┌──▼──┐ ┌──▼──┐ ┌───▼──┐ ┌──▼──┐ ┌───▼──┐ ┌──▼────────┐
│数据 │ │策略 │ │采样  │ │回测 │ │评估  │ │反思        │
│层   │ │层   │ │层    │ │引擎 │ │层    │ │层          │
└─────┘ └─────┘ └──────┘ └─────┘ └──────┘ └───────────┘
```

### 六层架构

| 层级 | 职责 |
|------|------|
| **数据层** | 多数据源接入（yfinance、akshare 或 daily_stock_analysis 插件） |
| **策略层** | 双重表示：人类可读描述 + 可执行 YAML DSL |
| **采样层** | 按市场环境、风格、策略适用范围智能采样 |
| **回测引擎** | 信号级模拟，支持滑点和手续费 |
| **评估层** | 多维指标 + 防过拟合检测 |
| **反思层** | LLM 失败归因 + 以 LLM 为主、参数搜索为兜底的进化链路 |

## 🧬 进化流程示例

拿 **2026年4月10日** yfinance 实盘数据 + `--method llm` 的真实结果举例：

```
第 1 轮: rsi_reversion_v1
         0 个信号，置信度 8.1% — 策略基本废了

         LLM 看完回测结果说："入场条件互相冲突，太严。"
         改了: entry.logic and→or，RSI 30→35

第 2 轮: rsi_reversion_v2 — 能出信号了
         522 个信号，胜率 52.7%，平均 +0.96%
         置信度 39.2%

         LLM 又说："OR 逻辑虽然修好了，但噪声信号太多。"
         改了: RSI 35→32，量比 1.3→1.15，止损 pct→atr

第 3 轮: rsi_reversion_v3 — 最终冠军
         498 个信号，胜率 52.6%，平均 +1.22%
         置信度 56.3%（从 8.1% 到这，涨了 48 个百分点）
```

> 以上是真实运行结果，不是合成数据，不是启发式 fallback。完整报告建议在本地用 `alphaevo evolve <strategy_id> --output ...` 生成。

## 📋 策略 DSL 示例

AlphaEvo 使用 YAML DSL 定义策略，兼具可读性和可执行性：

```yaml
meta:
  id: trend_pullback_rebound_v3
  name: 强趋势回踩放量反包
  version: 3
  parent_id: trend_pullback_rebound_v2
  market: a_share
  category: trend
  tags: [趋势, 回踩, 放量]

description: |
  适用于趋势市。个股近20日相对强于行业指数，
  5日线在10日线上方，回踩10日线附近但未破20日线，
  当日放量1.5倍以上。

entry:
  logic: and
  triggers:
    - indicator: relative_strength_20d
      op: ">"
      value: 0.12
  guards:
    - indicator: ma5_above_ma10
      op: "=="
      value: true
    - indicator: volume_ratio_1d_5d
      op: ">"
      value: 1.5

exit:
  triggers:
    - indicator: close_below_ma10
      op: "=="
      value: true
  stop_loss:
    type: atr
    atr_period: 21
    multiplier: 2.0
  take_profit:
    type: rr
    value: 2.0
  max_holding_days: 10

params:
  tunable:
    - target: entry.triggers[indicator=relative_strength_20d].value
      range: [0.05, 0.20]
      step: 0.01
```

完整的 DSL 规范请参考 [AGENTS.md §7](../AGENTS.md)。

## 📊 CLI 命令一览

```bash
# 初始化
alphaevo init                         # 交互式初始化配置

# 策略管理
alphaevo strategy create              # 交互式创建策略（需要 LLM）
alphaevo strategy draft "<idea>"      # 一句话策略生成可执行 YAML（无需 LLM）
alphaevo strategy research "<idea>"   # 一句话策略→保存→回测→入口/退出优化（无需 LLM）
alphaevo strategy revise <id> "<改法>" # 对已有策略做规则化修订（无需 LLM）
alphaevo strategy improve <id> "<改法>" # 修订已有策略→保存→回测→建议/可选优化（无需 LLM）
alphaevo strategy list                # 列出所有策略
alphaevo strategy show <id>           # 查看策略详情
alphaevo strategy import <file>       # 导入策略 YAML 文件
alphaevo strategy validate <file>     # 验证策略格式
alphaevo strategy diff <id1> <id2>    # 比较两个策略 DSL 差异

# 核心闭环
alphaevo run <strategy_id>            # 运行完整闭环（采样→回测→评估→报告）
alphaevo optimize <strategy_id>       # 优化买点阈值、指标周期、卖点/风控规则
  --spaces entry,params,indicator,exit,stoploss,takeprofit,holding,all
  --objective quality                 #   可选: confidence / win_rate / avg_return / drawdown / quality / profit_quality / robust_profit_quality
  --min-win-rate 0.5                  #   50% 胜率门槛
  --min-avg-return 0                  #   过滤负期望候选
  --min-total-return 0.10             #   过滤总收益不足的候选
  --min-profit-loss-ratio 1.0         #   过滤盈亏比过低的候选
  --max-drawdown 0.35                 #   最大回撤门槛
  --min-signals 30                    #   最小信号数门槛
  --param-max-changes 2               #   参数组合搜索深度
  --max-values-per-param 8            #   每个参数最多测试的候选值
  --evaluation-mode fast              #   候选评估: fast / full
  --full-eval-top 5                   #   fast 模式下完整复评前 N 名
  --reject-overfit                    #   拒绝完整评估中被标记过拟合的候选
  --max-train-val-gap 0.12            #   训练/验证胜率落差上限
  --max-val-test-gap 0.10             #   验证/测试胜率落差上限
  --max-walk-forward-gap 0.12         #   walk-forward 平均训练/测试落差上限
  --min-walk-forward-pass-rate 0.5    #   walk-forward 通过折数比例下限
  --parallel-workers 4                #   并发评估候选
  --joint-top 3                       #   前 N 个买点/参数候选继续做卖点联合搜索
  --joint-candidates-per-seed 64      #   每个联合搜索种子的卖点候选预算
  --fill-policy conservative          #   同 K 线止损/止盈冲突处理
alphaevo evolve <strategy_id>         # 进化策略（需要 LLM）
  --rounds 3                          #   进化轮数
  --method llm                        #   进化方法: llm / param_search / hybrid

# 评估与排行
alphaevo leaderboard                  # 策略排行榜
alphaevo compare <id1> <id2>          # 策略对比

# 进化树
alphaevo tree <strategy_id>           # 查看策略进化树
alphaevo tree --all                   # 全局进化树

# 配置
alphaevo config show                  # 查看当前配置
alphaevo config set <key> <value>     # 设置配置项

# 其他
alphaevo demo                         # 用内置数据跑完整 demo（无需网络/API Key）
alphaevo demo --real                  # 无需 LLM 的真实数据 demo
alphaevo factor discover <symbol>     # LLM 驱动因子发现
alphaevo version                      # 显示版本号
```

## 📚 参考

- **FunSearch (Nature 2024)** — Island 并行搜索
- **OPRO (DeepMind 2023)** — LLM 当优化器
- **Voyager (2023)** — 技能库 + 长期记忆

AlphaEvo 把这些思路落到了量化策略研究这个具体场景。

## 🔬 真实验证结果（2026 年 4 月 10 日）

故意同时放成功和失败的案例——靠谱的工具应该在证据不够时老实停下来。

### 基线回测

| 策略 | 信号数 | 平均收益 | 置信度 |
|------|--------|----------|--------|
| `ma_crossover_v1` | 54 | -0.26% | **24.3%** |
| `rsi_reversion_v1` | 0 | 0.00% | **8.1%** |

### LLM 进化结果

| 策略 | 起点 | 最优 | 发生了什么 |
|------|------|------|------|
| `rsi_reversion_v1` | 8.1% | 56.3% | 3 轮内从零信号修成可交易策略 |
| `ma_crossover_v1` | 24.2% | 24.2% | LLM 改了，但没通过防过拟合检验，保留基线 |

### 同日失败案例

| 策略 | 起点 | 发生了什么 |
|------|------|------------|
| `rsi_reversion_v1` smoke | 7.7% | 扩样后仍只有 1/30 个有效信号，停止进化 |
| `ma_crossover_v1` smoke | 15.2% | LLM 给了 v2，anti-overfit 拒绝晋升 |
| `sector_rotation_leader_v1` smoke | 11.3% | 候选版本看着更好但 `train_val_gap=18.9%`，判为过拟合 |

### 因子发现

`alphaevo factor discover AAPL` 真实结果（3 个提案 → 2 个通过验证并注册）：
- `volatility_compressed_breakout_quality` — IC Mean `0.306`，IR `1.76`
- `volume_confirmed_reversal` — IC Mean `0.248`，IR `1.50`

详情：[Factor Discovery Walkthrough](cookbook/07_factor_discovery_walkthrough.md)

### 导出产物

每次 evolve 会自动导出研究报告、LLM 证据日志、research log 和 trajectory 数据。

### 可复跑 Benchmark

```bash
python scripts/experiments/run_repro_benchmark.py \
  --adapter yfinance --method llm --rounds 3 \
  --output results/repro-benchmark/
```

详细记录：[2026-04-10 真实 LLM 验证报告](reports/2026-04-10-real-llm-validation_CN.md)

### 已知边界

- 外部 LLM 服务可能有延迟和 timeout，重复跑结果不会完全一致
- LLM 诊断合理 ≠ 一定过得了防过拟合检验（`ma_crossover_v1` 就是例子）
- 自动发现的因子是研究候选，正式使用前建议人工复核

## 🧠 Trajectory 数据飞轮

每次进化不光改策略，还自动生成训练数据：

- `trajectory.jsonl`：每轮（状态 → 诊断 → 改动 → 结果）的结构化记录
- `sharegpt.jsonl`：对话格式，可直接拿去做 SFT
- `preference.jsonl`：优劣对比，适合 DPO 训练

也就是说，跑得越多，积累的"怎么做策略研究"的数据就越多。

详见：[Trajectory Data Flywheel](trajectory_data_flywheel.md)

## 🛡️ 防过拟合

AlphaEvo 内置多重防过拟合机制：

- **时间分离** — 训练 / 验证 / 测试区间严格分开
- **Walk-Forward** — 滚动 12 个月训练 → 1 个月测试的前进窗口
- **复杂度惩罚** — 条件越多，评分扣得越重
- **稳定性检查** — 跨年份/行业/市场环境表现必须一致
- **最小信号数** — 信号 < 30 的策略不进排行榜
- **参数敏感度** — 参数扰动 ±10% 后性能衰减 > 30% 则警告

## 🔌 兼容性

### 独立模式

开箱即用，内置 `yfinance`（美股/港股）和 `akshare`（A 股）数据源。

### 插件模式

可对接 [daily_stock_analysis](https://github.com/ZhuLinsen/daily_stock_analysis)，复用其多数据源自动 fallback 能力。

```python
# 独立模式
from alphaevo.data.adapters.yfinance import YFinanceAdapter
data_manager = DataManager([YFinanceAdapter()])

# 插件模式
from alphaevo.data.adapters.dsa import DSAAdapter
data_manager = DataManager([DSAAdapter(dsa_path="/path/to/dsa")])
```

## 📦 安装选项

```bash
# 核心安装（仅回测，无需 LLM）
pip install alphaevo

# 带 LLM 支持（策略生成和进化功能）
pip install "alphaevo[llm]"

# 默认美股 / 港股 / 部分 A 股数据源（yfinance）
pip install "alphaevo[data-yfinance]"

# A 股数据源
pip install "alphaevo[data-akshare]"

# 全部数据源
pip install "alphaevo[data-full]"

# 开发环境
pip install "alphaevo[dev]"
```

## 🗺️ 路线图

参考同类项目后的优化分析见：[competitive_optimization_analysis.md](competitive_optimization_analysis.md)。

- [x] Phase 1: 策略研究闭环 (MVP) — 回测引擎、指标、评估器
- [x] Phase 2: 自我进化流水线 — LLM 反思、变异、多轮改进
- [x] Phase 3: CLI & 编排 — 完整命令行、策略存储、排行榜
- [x] Phase 4: 开源打磨 — CI/CD、文档、英文模板、CHANGELOG
- [x] Phase 5: 真实验证 — 实盘数据回测 + 真实 LLM 进化验证
- [ ] Phase 6: 进化经验库 — 跨策略经验复用、进化记忆
- [ ] Phase 7: 市场环境自适应
- [ ] Phase 8: Web UI 仪表盘

## 🤝 参与贡献

欢迎贡献！请查看 [CONTRIBUTING.md](../CONTRIBUTING.md) 了解详细指南。

**特别欢迎以下贡献：**
- 新的策略模板
- 数据源适配器
- 评估指标
- UI / 可视化改进

## ☕ 支持与联系

觉得有用的话给个 ⭐ 吧！

<table>
  <tr>
    <td align="center" width="220">
      <a href="https://www.xiaohongshu.com/user/profile/61594417000000000201fa68" target="_blank">
        <img src="../sources/xiaohongshu.png" width="160" alt="小红书"><br>
        <b>小红书 📱</b>
      </a><br>
      <sub>关注获取量化策略研究动态</sub>
    </td>
    <td valign="top" style="padding-left: 24px">
      <b>📬 联系与合作</b><br><br>
      🐛 &nbsp;<a href="https://github.com/ZhuLinsen/alphaevo/issues">提交 Issue</a> — Bug 反馈 / 功能建议<br>
      📧 &nbsp;<a href="mailto:zhuls345@gmail.com">zhuls345@gmail.com</a> — 商务合作<br>
      🔗 &nbsp;<a href="https://github.com/ZhuLinsen/daily_stock_analysis">daily_stock_analysis</a> — 关联项目， AI 驱动的每日股票智能分析系统
    </td>
  </tr>
</table>

## ⚠️ 免责声明

**本项目仅供研究和学习，不是投资建议。**

作者不对使用本软件导致的任何经济损失负责。投资决策前请自行研究并咨询专业人士。

回测结果是模拟计算，不代表未来表现。

## 📄 许可证

Apache-2.0 License — 详见 [LICENSE](../LICENSE)。

如果你在项目中用到或基于它做二次开发，欢迎注明来源。

---

<div align="center">


</div>
