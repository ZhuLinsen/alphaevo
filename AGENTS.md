# AGENTS.md

本文件是 **AlphaEvo** 仓库的 AI 协作规则与项目规范唯一真源。

---

## 1. 项目定位

**AlphaEvo — Open-Source RSI (Recursive Self-Improvement / Evo) Engine for Stock Research** — 开源股票策略、因子与研究机制的三层递归自我进化引擎。

核心定位：**面向股票投研的递归自我进化（Stock RSI）引擎** — 用不可变 L0 真相核与双时态前瞻影子账本约束 AI 研究闭环：「历史回测只负责否决，前瞻样本外影子账本才决定晋升」。

**不承诺收益神话，强调研究闭环。**

### 核心闭环

```
提出策略假设 → 自动选择样本 → 历史/滚动验证 → 输出多维评估
→ 失败归因 → 自动改写策略 → 再次测试 → 策略进化树
```

---

## 2. 仓库结构

### 架构七层

| 层级 | 目录 | 职责 |
|------|------|------|
| **Data Layer** | `src/alphaevo/data/` | 多数据源接入，统一 MarketSnapshot 输出 |
| **Strategy Layer** | `src/alphaevo/strategy/` | 策略 DSL 定义、解析、序列化、版本管理 |
| **Sampler Layer** | `src/alphaevo/sampler/` | 分层采样（市场环境/风格/策略适用范围） |
| **Backtest Layer** | `src/alphaevo/backtest/` | 指标注册表 + 条件评估 + 回测引擎 + 组合回测 |
| **Evaluator Layer** | `src/alphaevo/evaluator/` | 多维评估（胜率/盈亏比/回撤/环境适应性） |
| **Reflection Layer** | `src/alphaevo/reflection/` | 失败归因 + 修正建议 + 下一版策略生成 |
| **Orchestrator** | `src/alphaevo/orchestrator/` | 端到端流程编排 |

### 辅助模块

| 模块 | 目录 | 职责 |
|------|------|------|
| **Models** | `src/alphaevo/models/` | Pydantic v2 数据模型 |
| **CLI** | `src/alphaevo/cli/` | Typer + Rich 命令行界面 |
| **Config** | `src/alphaevo/core/config.py` | 统一配置管理 (Pydantic Settings) |
| **Leaderboard** | `src/alphaevo/leaderboard/` | 策略排行榜，综合评分 |
| **Utils** | `src/alphaevo/utils/` | 公共工具函数 (utcnow, fmt_pct, safe_div 等) |
| **Alpha Factory** | `src/alphaevo/alpha_factory/` | LLM 驱动因子合成、验证、注册 |
| **Research Log** | `src/alphaevo/research_log/` | 进化过程结构化记录与渲染 |
| **Experience Store** | `src/alphaevo/reflection/experience.py` | 进化经验持久化 (SQLite)，跨策略学习 |
| **Self-Critic** | `src/alphaevo/reflection/critic.py` | 变更质量门控 + 多候选排名 |
| **Meta-Learner** | `src/alphaevo/reflection/meta_learner.py` | 数据驱动的进化策略自适应 |
| **Pattern Library** | `src/alphaevo/strategy/library/` | 跨策略可复用模式提取与注入 |
| **L0 Truth Kernel** | `src/alphaevo/truth/` | TrialLedger、Deflated Sharpe、两层 Null Control、运行时 PIT 未来噪声探针 |
| **Forward Shadow Ledger** | `src/alphaevo/forward/` | 双时态 `as_of` 提交、SHA-256 哈希链、T+5/10/20 归因与非对称晋升/熔断状态机 |
| **RSI Engine (L1/L2/L3)** | `src/alphaevo/rsi/` | `program.md` 驱动棘轮、MAP-Elites、Co-STEER 盲区路由、异构模型议会、机制 AB 进化 |

### 关键入口

| 文件 | 用途 |
|------|------|
| `src/alphaevo/cli/main.py` | CLI 入口 |
| `src/alphaevo/strategy/dsl/parser.py` | 策略 YAML 解析器 |
| `src/alphaevo/data/adapter.py` | 数据适配器抽象接口 + DataManager |
| `src/alphaevo/models/` | Pydantic 数据模型 |
| `strategies/builtin/` | 内置策略模板 (YAML) |
| `tests/unit/` | 单元测试 |
| `src/alphaevo/core/config.py` | 统一配置系统 |
| `src/alphaevo/strategy/dsl/serializer.py` | 策略序列化 (Strategy → YAML) |
| `src/alphaevo/backtest/indicators.py` | 指标注册表 + 条件评估器 |
| `src/alphaevo/backtest/engine.py` | 回测引擎 |
| `src/alphaevo/orchestrator/pipeline.py` | 主流程编排 |

---

## 3. 硬规则

### 代码规范

- 遵循现有目录边界，不跨层放置逻辑。
- 新模块必须有对应的 `__init__.py` 和类型导出。
- 所有公共接口使用 Pydantic v2 模型，不用裸 dict。
- 策略定义必须同时有**人类可读描述**和**可执行 DSL**。
- 不写死密钥、路径、模型名、端口。所有可配置项通过 `AppConfig` 或环境变量读取。
- 复用 daily_stock_analysis 的数据能力时通过 adapter 层隔离，不直接依赖其内部实现。
- 时间相关默认值统一使用 `datetime.now(timezone.utc)` (UTC)。

### Git 规范

- 未经确认，不执行 `git commit`、`git tag`、`git push`。
- Commit message 使用 Conventional Commits 格式：`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`。
- PR 必须包含：改了什么、为什么、验证情况、风险点。

### 安全规范

- 不承诺投资收益，README 和所有输出必须包含免责声明。
- 不存储用户真实交易数据，仅处理公开市场数据。
- API Key 等敏感信息只通过环境变量传入，不落配置文件。

---

## 4. 技术栈

### 核心依赖（pip install alphaevo 即装）

| 技术 | 用途 | 版本要求 |
|------|------|----------|
| Python | 运行时 | >= 3.10 |
| Pydantic | 数据模型 & 校验 | >= 2.0 |
| Typer | CLI 框架 | >= 0.9 |
| Rich | CLI 美化 | >= 13.0 |
| pandas | 数据处理 | >= 2.0 |
| PyYAML | 策略 DSL 解析/序列化 | >= 6.0 |
| tenacity | 重试逻辑 | >= 8.2 |
| SQLite | 策略版本 & 评估存储 | 内置 |

### 可选依赖（按需安装）

| 安装方式 | 包含 | 用途 |
|----------|------|------|
| `pip install alphaevo[llm]` | litellm >= 1.40 | LLM 驱动的策略生成/反思（进化功能必需） |
| `pip install alphaevo[data-yfinance]` | yfinance >= 0.2 | 美股/港股/A股(有限)数据源 |
| `pip install alphaevo[data-akshare]` | akshare >= 1.12 | A 股全量数据源 |
| `pip install alphaevo[data-full]` | yfinance + akshare + efinance + tushare + baostock | 全部数据源 |
| `pip install alphaevo[tui]` | textual >= 0.40 | TUI 升级 |
| `pip install alphaevo[charts]` | plotext >= 5.0, matplotlib >= 3.7 | 终端/图形图表 |
| `pip install alphaevo[dev]` | pytest, ruff, mypy, pre-commit | 开发工具 |

> **设计决策**: `litellm` 从核心依赖移至 optional。纯回测场景（无 LLM 进化）不需要 litellm 及其 200+ MB 子依赖。`alphaevo run` 和 `alphaevo leaderboard` 仅需核心依赖即可运行；`alphaevo evolve` 和 `alphaevo strategy create` 需要 `[llm]`。

---

## 5. 配置管理

### 配置优先级（高 → 低）

1. CLI 参数（`--model`, `--adapter` 等）
2. 环境变量（`ALPHAEVO_LLM_MODEL`, `ALPHAEVO_DATA_ADAPTER` 等）
3. 项目配置文件（`.alphaevo/config.yaml`，在项目根目录）
4. 用户配置文件（`~/.alphaevo/config.yaml`）
5. 内置默认值

### 关键配置项

| 配置项 | 环境变量 | 默认值 | 说明 |
|--------|----------|--------|------|
| `llm.model` | `ALPHAEVO_LLM_MODEL` | `gemini/gemini-3.8-flash` | 策略生成/反思用的 LLM |
| `llm.reflect_model` | `ALPHAEVO_LLM_REFLECT_MODEL` | 空（复用 llm.model） | 反思专用模型 |
| `llm.api_key` | `ALPHAEVO_API_KEY` | — | LLM API Key（**仅通过环境变量**） |
| `data.adapter` | `ALPHAEVO_DATA_ADAPTER` | `auto` | 数据源：auto / tencent / yfinance / akshare / dsa；auto 为 dsa(可选) → tencent → akshare → yfinance |
| `data.cache_dir` | `ALPHAEVO_CACHE_DIR` | `~/.alphaevo/cache/` | 数据缓存目录 |
| `data.source_failure_threshold` | `ALPHAEVO_DATA_SOURCE_FAILURE_THRESHOLD` | `3` | 单数据源连续失败达到阈值后临时熔断 |
| `data.source_cooldown_seconds` | `ALPHAEVO_DATA_SOURCE_COOLDOWN_SECONDS` | `60.0` | 数据源熔断冷却秒数 |
| `data.max_concurrent_fetches` | `ALPHAEVO_DATA_MAX_CONCURRENT_FETCHES` | `8` | 异步并发抓取最大并发数 |
| `db.path` | `ALPHAEVO_DB_PATH` | `~/.alphaevo/alphaevo.db` | SQLite 数据库路径 |
| `backtest.slippage` | — | `0.001` | 默认滑点 |
| `backtest.commission` | — | `0.0003` | 默认手续费率 |
| `backtest.fill_policy` | `ALPHAEVO_BACKTEST_FILL_POLICY` | `conservative` | 同 K 线止损/止盈冲突处理：conservative / optimistic / close_first |
| `backtest.walk_forward_folds` | `ALPHAEVO_BACKTEST_WALK_FORWARD_FOLDS` | `3` | Walk-Forward 默认折数 |
| `backtest.walk_forward_train_pct` | `ALPHAEVO_BACKTEST_WALK_FORWARD_TRAIN_PCT` | `0.7` | Walk-Forward 训练集比例 |
| `backtest.walk_forward_pass_gap` | `ALPHAEVO_BACKTEST_WALK_FORWARD_PASS_GAP` | `0.10` | Walk-Forward 单折通过阈值 |
| `backtest.stress_window_days` | `ALPHAEVO_BACKTEST_STRESS_WINDOW_DAYS` | `20` | Stress-window 评测窗口长度 |
| `backtest.stress_window_top_k` | `ALPHAEVO_BACKTEST_STRESS_WINDOW_TOP_K` | `3` | Stress-window 最差窗口数量 |
| `evolution.max_rounds` | — | `5` | 进化最大轮数 |
| `evolution.max_changes_per_round` | — | `3` | 每轮最大修改数 |
| `evolution.num_candidates` | — | `3` | LLM 每轮生成候选实验数量 |
| `rsi.shadow_ledger_dir` | `ALPHAEVO_SHADOW_LEDGER_DIR` | `~/.alphaevo/shadow_ledger` | 双时钟前瞻影子账本目录 |
| `rsi.proposer_model` | `ALPHAEVO_RSI_PROPOSER_MODEL` | `None` | RSI 异构议会 Proposer 模型（如 `openai/gpt-6-astra`） |
| `rsi.prosecutor_model` | `ALPHAEVO_RSI_PROSECUTOR_MODEL` | `None` | RSI 异构议会 Prosecutor 模型（如 `anthropic/claude-opus-5.5`） |
| `rsi.null_permutations` | `ALPHAEVO_RSI_NULL_PERMUTATIONS` | `100` | RSI Null 对照置换次数 |

### `alphaevo init` 命令

首次使用交互式引导：

```bash
alphaevo init
# → 选择默认数据源
# → 选择 LLM 模型（可跳过）
# → 创建 ~/.alphaevo/config.yaml
# → 初始化 SQLite 数据库
```

---

## 6. 兼容性设计

### 独立模式
AlphaEvo 可完全独立运行，内置轻量数据获取能力（通过 auto / tencent / yfinance / akshare）。

### 插件模式
通过 adapter 对接 daily_stock_analysis 的 `DataFetcherManager`，复用其多数据源 fallback 能力。

```python
# 独立模式
from alphaevo.core.config import AppConfig
from alphaevo.data.adapter import DataManager, get_adapter_chain
data_manager = DataManager(get_adapter_chain(AppConfig()))

# 插件模式
from alphaevo.data.adapters.dsa import DSAAdapter
data_manager = DataManager([DSAAdapter(dsa_path="/path/to/dsa")])
```

`auto` 数据链是当前默认：`dsa`（可选，配置了 `data.dsa_path` 才启用）→ `tencent` → `akshare` → `yfinance`。
`tencent` 是轻量直接 K 线/实时行情源，优先用于 A 股 OHLCV 稳定性；`DSA/AkShare` 继续承担更丰富的股票池、行业/事件上下文。
`DataManager.health_status()` 返回各数据源 success/failure/consecutive_failures/disabled/last_error/recent_errors，供 run/evolve 诊断和报告使用。

### 数据质量优先进化规则

LLM 反思/进化前必须先检查数据质量：当事件/新闻指标主要来自 proxy、provider coverage 很低、或数据源连续失败时，优先触发 `data_quality` playbook，先诊断数据链路与 proxy 可靠性，再决定是否修改策略规则。不要让 LLM 对低质量或 proxy-only 数据过度调参。

> **目录约定**: 所有具体数据适配器放在 `src/alphaevo/data/adapters/`，不放顶层 `src/alphaevo/adapters/`（该目录废弃）。

### 策略兼容
AlphaEvo 的策略 DSL 是 daily_stock_analysis 自然语言策略的**超集**。可导入现有策略并通过 LLM 自动补全 DSL 字段。

---

## 7. 策略 DSL 规范 (v0.5)

策略分两层表示：

### A. 人类可读层 (`description`)
自然语言描述，方便 LLM 生成、解释、传播。

### B. 可执行层 (YAML DSL)
YAML 结构化定义，用于自动回测和参数搜索。

```yaml
meta:
  id: trend_pullback_rebound_v1
  name: 强趋势回踩放量反包
  version: 1
  parent_id: null            # 进化树父节点
  created_at: "2026-03-31"
  market: a_share
  category: trend            # trend / reversal / event / rotation / framework
  tags: [趋势, 回踩, 放量]
  preferred_regime:          # v0.3 新增：适用市场环境（可选，用于环境 gating）
    - trending_up

description: |
  适用于趋势市。个股近20日相对强于行业指数，5日线在10日线上方，
  回踩10日线附近但未破20日线，当日放量1.5倍以上，近期无重大利空。

universe:
  market: [a_share_main]
  filters:
    - field: market_cap
      op: ">="
      value: 5_000_000_000     # 50亿以上

entry:
  logic: and                   # 条件组合逻辑: and | or (默认 and)
  triggers:                    # v0.5 新增：真正触发买入的信号；为空时回退到 conditions
    - indicator: volume_ratio_1d_5d
      op: ">"
      value: 1.5
  guards:                      # v0.5 新增：硬过滤条件，始终 AND；为空时只使用 filters
    - indicator: relative_strength_20d
      op: ">"
      value: 0.08
  conditions:                  # 兼容字段：旧版策略仍可把买入条件写在这里
    - indicator: ma5_above_ma10
      op: "=="
      value: true
    - indicator: close_to_ma10_pct
      op: "<="
      value: 0.015
  filters:                     # filters 始终为 AND (全部满足)
    - indicator: negative_news_score
      op: "<"
      value: 0.4
  execution:                   # v0.3 新增：入场执行方式
    timing: next_open          # next_open | close | breakout_high (默认 next_open)
    slippage: 0.001            # 滑点比例 (默认从 config 读取)

exit:
  triggers:                     # v0.4 新增：显式卖出/退出触发器，任一满足即按 signal 退出
    - indicator: close_below_ma10
      op: "=="
      value: true
  stop_loss:
    type: pct                  # pct | atr | price_level | pct_from_low | composite
    value: 0.04
  take_profit:
    type: rr                   # rr | pct | target_ma | trailing
    value: 2.0
  max_holding_days: 10

market_rules:                  # 市场特殊规则 (按 meta.market 自动应用)
  a_share:
    t_plus_1: true
    limit_up_down: true
    suspension: true

params:
  tunable:
    - target: entry.guards[indicator=relative_strength_20d].value
      range: [0.05, 0.20]
      step: 0.01
    - target: entry.triggers[indicator=volume_ratio_1d_5d].value
      range: [1.2, 2.5]
      step: 0.1
    - target: exit.stop_loss.value
      range: [0.02, 0.08]
      step: 0.01
    - target: entry.conditions[indicator=close_to_ma10_pct].indicator
      range: [5, 20]
      step: 5
    - target: exit.take_profit.target
      range: [20, 80]
      step: 5
```

> `params.tunable.target` 除了 `...value` 以外，也支持调指标周期本身：
> `entry.triggers[...].indicator` / `entry.guards[...].indicator` /
> `entry.conditions[...].indicator` / `entry.filters[...].indicator` /
> `entry.triggers[...].indicator.fast` / `entry.triggers[...].indicator.slow` /
> `entry.conditions[...].indicator.fast` / `entry.conditions[...].indicator.slow` /
> `entry.conditions[...].indicator.signal` /
> `entry.conditions[...].indicator.std` /
> `entry.filters[...].indicator.fast` / `entry.filters[...].indicator.slow` /
> `entry.filters[...].indicator.signal` /
> `entry.filters[...].indicator.std` /
> `exit.take_profit.target` / `exit.stop_loss.atr_period` /
> `exit.max_holding_days`。
> 单一 `...indicator` target 目前支持常见窗口型指标，例如
> `close_above_ma60`、`close_to_ma20_pct`、`ma20_slope`、`rsi_14`、
> `atr`、`bollinger_band_width`、`price_above_bollinger_upper`、
> `price_below_bollinger_lower`、`volume_ratio_1d_5d`、`momentum_10d`、
> `avg_volume_20d`、`breakout_high_20d`、`price_position_120d`、
> `days_since_high_20d`、`days_since_low_20d`、`volatility_20d`、
> `relative_strength_20d`，以及 `target_ma: ma60`；
> 双均线指标则支持分别调快线/慢线，例如
> `entry.conditions[indicator=ma5_ge_ma10_or_crossing].indicator.fast` /
> `entry.conditions[indicator=ma5_ge_ma10_or_crossing].indicator.slow`。
> `MACD` 指标则支持分别调 `fast/slow/signal`，例如
> `entry.conditions[indicator=macd_histogram].indicator.fast` /
> `entry.conditions[indicator=macd_histogram].indicator.slow` /
> `entry.conditions[indicator=macd_histogram].indicator.signal`。
> `Bollinger` 指标则支持分别调窗口和标准差倍数，例如
> `entry.conditions[indicator=bollinger_band_width].indicator` /
> `entry.conditions[indicator=bollinger_band_width].indicator.std`。
>
> `stop_loss.type: atr` 时可选 `atr_period` 字段；省略时默认使用 `atr(14)`。

### DSL 变更记录

| 版本 | 变更 |
|------|------|
| v0.1 | 初始 DSL：meta/description/universe/entry/exit/params |
| v0.2 | `entry.logic` 支持 and/or；`params.tunable` 改用指标名做 key；新增 `market_rules` |
| v0.3 | 新增 `entry.execution`（入场时机+滑点）；新增 `meta.preferred_regime`；`composite` 退出条件标准化为 `StrategyCondition` 结构 |
| v0.4 | 新增 `exit.triggers` 显式卖出触发器；回测命中时以 `ExitReason.SIGNAL` 按当前 close 平仓；新增 `alphaevo optimize` 做入口阈值、指标周期、退出/风控参数搜索 |
| v0.5 | 新增 `entry.triggers` / `entry.guards`，将真正买点与硬过滤条件拆开；新增 `backtest.fill_policy` 处理同 K 线止损/止盈冲突 |

### 入场触发与过滤 (v0.5)

`entry.triggers` 表示真正触发买入的信号，`entry.guards` 表示硬过滤条件。旧版 `entry.conditions` / `entry.filters` 继续兼容：

- 若 `entry.triggers` 非空，买点由 `triggers` 按 `entry.logic` 组合判断。
- 若 `entry.triggers` 为空，则回退到旧版 `entry.conditions`。
- `entry.guards` 与 `entry.filters` 始终按 AND 组合。

这能避免把“突破/反包/金叉”等触发信号和“趋势向上/非 ST/低波动”等过滤条件混在一起。

### 显式卖出触发器 (v0.4)

`exit.triggers` 使用与 `entry.conditions` 相同的 `StrategyCondition` 结构，但语义是**持仓期间任一触发即退出**。它用于表达用户明确说出的卖点，例如跌破均线、RSI 过热、趋势破坏。

执行优先级：

1. `max_holding_days`
2. `stop_loss`
3. `take_profit`
4. `exit.triggers`（按当前 close 退出，`exit_reason=signal`）

同一根 K 线若同时触发止损和止盈，使用 `backtest.fill_policy`：

- `conservative`：默认，按止损先成交。
- `optimistic`：按止盈先成交。
- `close_first`：按当日收盘相对入场价方向判定。

```yaml
exit:
  triggers:
    - indicator: close_below_ma10
      op: "=="
      value: true
    - indicator: rsi_14
      op: ">"
      value: 75
  stop_loss:
    type: pct
    value: 0.04
  take_profit:
    type: rr
    value: 2.0
  max_holding_days: 10
```

### composite 退出条件标准化 (v0.3)

composite 类型的 stop_loss.conditions 必须使用与 entry.conditions 相同的 `StrategyCondition` 结构：

```yaml
exit:
  stop_loss:
    type: composite
    conditions:
      - indicator: sector_heat_rank    # 板块热度退出前10则止损
        op: ">"
        value: 10
      - indicator: close_below_ma10    # 跌破MA10则止损
        op: "=="
        value: true
```

> ⚠️ 不再使用 `{type: sector_rank_exit, threshold: 10}` 等自定义格式。

---

## 8. 内置策略

| 策略文件 | 类别 | 说明 |
|----------|------|------|
| `trend_pullback_rebound.yaml` | trend | 核心指标基于 OHLCV；news 类过滤使用价格/量能 proxy |
| `mean_reversion_oversold.yaml` | reversal | 所有指标基于 OHLCV |
| `event_driven_breakout.yaml` | event | 依赖 proxy 指标（`news_sentiment_score` 等），标记 `experimental` |
| `sector_rotation_leader.yaml` | rotation | 板块数据走 context，资金流/事件类走 proxy，标记 `experimental` |
| `ma_crossover.yaml` | trend | 双均线交叉，纯 OHLCV |
| `rsi_reversion.yaml` | reversal | RSI 超卖反转，纯 OHLCV |
| `volatility_compression_breakout.yaml` | trend | 突破前高 + 波动收缩 + trailing 止盈，纯 OHLCV |

> `experimental` 策略的新闻/事件链路依赖 proxy，结果更适合作为研究参考而非最终结论。

---

## 9. 指标分层规范

### 第 1 层 — MVP (仅 OHLCV，可立即实现)

```
ma5_above_ma10, close_to_ma10_pct, close_above_ma20,
volume_ratio_1d_Nd, rsi_N, deviation_from_ma20_pct,
has_stop_signal, volume_shrink_then_rise,
ma5_ge_ma10_or_crossing, atr/atr_N, close_below_ma10,
macd_histogram/macd_histogram_fastN_slowM_signalK,
macd_cross_bullish/macd_cross_bullish_fastN_slowM_signalK,
bollinger_band_width/bollinger_band_width_Nd/bollinger_band_width_Nd_stdS,
price_above_bollinger_upper/price_above_bollinger_upper_Nd/price_above_bollinger_upper_Nd_stdS,
price_below_bollinger_lower/price_below_bollinger_lower_Nd/price_below_bollinger_lower_Nd_stdS,
breakout_high_Nd, price_position_Nd,
ma20_slope, momentum_Nd, avg_volume_Nd, consecutive_up_days,
days_since_high_Nd, days_since_low_Nd, rsi_N_zscore, volatility_Nd
```

### 第 2 层 — 需额外数据源

```
relative_strength_Nd (需基准指数)
st_flag (需股票标记)
sector_heat_rank, sector_heat_rising_days (需板块数据)
intra_sector_strength_rank_pct (需板块内排名)
```

### 第 3 层 — 需新闻/事件 API

```
negative_news_score, news_sentiment_score,
days_since_event, price_above_pre_event,
sector_fund_flow_positive, already_overreacted
```

### 降级 / 代理策略

L2/L3 指标采用分层兜底：

- 能从 benchmark / sector context 计算的，优先走真实上下文。
- 新闻 / 事件类指标在没有外部 feed 时，退化为价格缺口 + 量能放大 + 事件前价格锚点的 proxy。
- 若连 proxy 也无法稳定构造，则再返回中性默认值，确保回测流程不被阻断。

```python
@IndicatorRegistry.register("negative_news_score")
def negative_news_score(df, idx) -> float:
    return price_volume_event_proxy(df, idx).negative_score
```

---

## 10. 评估指标体系

### 基础指标

| 指标 | 说明 |
|------|------|
| win_rate | 胜率 |
| avg_return | 平均收益率 |
| median_return | 收益中位数 |
| profit_loss_ratio | 盈亏比 |
| max_drawdown | 最大回撤 |
| sharpe_ratio | 夏普比率 |
| signal_count | 信号总数 |
| avg_holding_days | 平均持仓天数 |
| max_consecutive_loss | 最大连续亏损次数 |
| total_return | 累计收益率 |

### 稳定性指标

| 指标 | 说明 |
|------|------|
| yearly_consistency | 年度一致性 (1 - std/mean) |
| regime_performance | 按市场环境分组表现 |
| regime_holdout_gap | 留一市场环境时的最差泛化落差 |
| param_sensitivity | 参数敏感度（扰动 ±10% 后性能衰减） |

### 综合评分公式

```
confidence_score =
    0.25 × win_rate_score         # 胜率 (70%=满分)
  + 0.15 × avg_return_score       # 平均收益 (5%=满分)
  + 0.15 × profit_loss_score      # 盈亏比 (2.5=满分)
  + 0.15 × drawdown_score         # 回撤 (30%=0分, 反转取值)
  + 0.10 × sharpe_score           # 夏普比率 (2.0=满分)
  + 0.10 × consistency_score      # 年度一致性
  + 0.10 × sensitivity_score      # 参数敏感度 (越低越好；未实测取中性值 0.5)
  - overfit_penalty               # train-val gap>10%: -0.15; val-test gap>8%: -0.10
  - complexity_penalty            # complexity_score × 0.10
```

> 当提供 `market_data` 时，回撤、Sharpe 与总收益主口径为按日等权 Mark-to-Market 组合序列（`metrics_basis="daily_mtm"`，`trade_sequence_*` 保留为诊断对照）。未显式运行参数扰动实测时，`sensitivity_score` 保持中性 `0.5`，不再默认给予 `1.0` 满分。
> ⚠️ 本公式与 `docs/module_tech_specs.md` §6 保持一致。修改时必须同步两处。

---

## 11. 防过拟合措施

1. **时间分离**: 训练区间 / 验证区间 / 测试区间严格分离
2. **Walk-Forward**: 滚动窗口验证（过去12月优化 → 测未来1月 → 滚动前进）
3. **Stress Window**: 基于 benchmark 最差窗口检验策略在高压区间的韧性
4. **极端行情独立测试**: 保留未见市场阶段
5. **复杂度惩罚**: 条件越多扣分越重，防止 LLM 堆条件
6. **稳定性要求**: 不同年份/行业/环境表现方差必须在阈值内
7. **最小信号数**: 信号数 < 30 的策略不参与排行
8. **参数敏感度**: 参数扰动 ±10% 性能衰减 > 30% 则警告

### 进化安全护栏

| 护栏 | 规则 | 目的 |
|------|------|------|
| 最大改动数 | 每轮最多改 3 个条件 | 防止 LLM 大幅跳变 |
| 复杂度上限 | 总条件 > 8 则强制简化 | 防止堆条件过拟合 |
| 必须变好 | confidence_score 不升则 prune | 防止随机游走 |
| 过拟合检测 | train-val gap > 15% 则停止 | 防止训练集过拟合 |
| 最小信号数 | < 30 个信号不算有效 | 防止小样本偏差 |
| 参数敏感度 | 参数扰动 ±10% 性能衰减 > 30% 则警告 | 防止参数脆弱 |

### Showcase / Provenance 规则

AlphaEvo 下一阶段优先服务 Star-facing Showcase Release：让新用户 30 秒内看到“初始策略失败 -> 失败归因 -> 结构化改写 -> 指标改善 -> 报告 / 进化树”的闭环。

Provenance 是 showcase 的可信背书，不是当前 P0 的完整平台主线。完整 artifact / official leaderboard 规范见 `docs/canonical_evaluation.md`，后续按路线图推进。

当前优先规则：

- `alphaevo demo` 和 flagship evolve 路径必须优先展示自我进化闭环，而不是只打印回测指标。
- README / demo / report 应突出 before/after、失败归因、mutation rationale、候选对比、top patterns、anti-overfit 提醒。
- 报告必须有最小 `Run Provenance` 区块：run id、strategy hash、data adapter/date range、config hash、dirty/data reproducibility 提示。
- 普通本地研究结果不得伪装成 official benchmark。
- `optimize` 报告至少展示搜索空间、objective、gates、候选数量、best candidate 来源。

后续完整规则：

- 每次运行生成唯一 `run_id` 或 `optimization_session_id`，格式优先采用 `YYYYMMDDTHHMMSSZ_<strategy_id>_<hash8>`。
- 本地 artifact 文件是唯一真源；SQLite 只做索引和查询，不复制完整 manifest / trades / evaluation。
- 工件目录默认使用：
  - `reports/runs/<run_id>/`
  - `reports/evolutions/<run_id>/`
  - `reports/optimizations/<session_id>/`
- 完整 artifact 阶段使用 JSON / JSONL / YAML / Markdown，不把 Parquet 或 pyarrow 放入 core 依赖；Parquet 只能作为 optional artifact extra。
- 每次运行必须保存 strategy snapshot、config snapshot、data fingerprint、evaluation、report、manifest。
- `config_snapshot.json` 使用结果相关字段白名单，不全量 dump 环境变量、数据库 URL、API key、token 或 proxy credential。
- data fingerprint 是硬要求；普通本地运行至少保存每个 symbol 的行数、起止日期、字段 hash、adapter、缓存命中状态。official 评测未来必须绑定不可变 `data_bundle_id`。
- 允许 dirty worktree 运行，但 manifest 必须记录 `code_dirty=true` 和 `diff_hash`。dirty run 不允许进入 official leaderboard。
- 评估协议分三档：
  - `fast_research_v1`: 本地快速迭代，允许 mutable provider、dirty code、较小样本。
  - `full_research_v1`: 完整本地研究，要求 train/val/test、walk-forward、stress-window、data fingerprint。
  - `official_v1`: 官方复评协议，要求 clean code、固定数据 bundle、固定市场/窗口/手续费/滑点、最小信号数、anti-overfit gates。
- 完整 artifact 阶段不实现完整 data bundle，但必须实现 official gate：不满足条件时拒绝或明确降级，并记录 `requested_protocol` 与 `effective_protocol`。
- run / evolve / optimize 都必须记录 `random_seed` 与 determinism policy。涉及 LLM 时记录 model/provider/temperature、prompt hash、response hash；完整 prompt/response 默认不进入 manifest，只能进入可选 trajectory artifact。
- `optimize` 必须保存完整搜索 provenance：搜索空间、预算、objective、gates、seed、parallel workers、所有候选摘要、best candidate 来源。不能只保存 best。
- Markdown report 必须包含简洁 `Run Provenance` 区块，至少显示 run id、protocol、strategy hash、code version/dirty、data reproducibility、adapter、date range、config hash、artifact dir。
- `leaderboard` 必须区分 local / official scope；不同协议的结果不能混排成一个 winner。
- `runs compare` 可以比较不同 protocol 的 run，但必须标记 `comparable=false`，只允许诊断差异，不给直接排名结论。

---

## 12. CLI 命令设计

```bash
# 初始化
alphaevo init                         # 交互式初始化配置

# 策略管理
alphaevo strategy create              # 交互式创建策略 (需要 [llm])
alphaevo strategy draft "<idea>"      # 从一句话策略生成可执行 YAML（无需 LLM）
alphaevo strategy research "<idea>"   # 一句话策略→保存→回测→入口/退出优化（无需 LLM）
alphaevo strategy revise <id> "<改法>" # 对已有策略做规则化修订（无需 LLM）
alphaevo strategy improve <id> "<改法>" # 修订已有策略→保存→回测→建议/可选优化（无需 LLM）
alphaevo strategy list                # 列出所有策略
alphaevo strategy show <id>           # 查看策略详情
alphaevo strategy diff <id1> <id2>    # 查看两个策略 DSL 差异
alphaevo strategy import <file>       # 导入策略 YAML 文件
alphaevo strategy validate <file>     # 验证策略 YAML 格式和条件
  --strict                            #   有 warning 也返回失败

# 因子管理
alphaevo factor list                  # 查看已发现因子
alphaevo factor show <name>           # 查看因子详情和代码
alphaevo factor retire <name>         # 退役一个因子
alphaevo factor discover <symbol>     # LLM 驱动因子发现 (需要 [llm])

# 核心闭环
alphaevo run <strategy_id>            # 运行完整闭环（采样→回测→评估→报告）
  --sampling strategy_scoped          #   覆盖采样模式
  --fill-policy conservative          #   止损/止盈同 K 线冲突策略
alphaevo optimize <strategy_id>       # 优化入口阈值、指标周期、退出/风控规则
  --spaces entry,params,indicator,exit,stoploss,takeprofit,holding,all
  --objective quality                 #   排序目标: confidence / win_rate / avg_return / drawdown / quality / profit_quality / robust_profit_quality
  --min-win-rate 0.5                  #   候选合格胜率门槛
  --min-avg-return 0                  #   候选合格平均收益门槛，避免刷胜率
  --min-total-return 0.10             #   候选合格复合总收益门槛
  --min-profit-loss-ratio 1.0         #   候选合格盈亏比门槛
  --max-drawdown 0.35                 #   候选合格最大回撤上限
  --min-signals 30                    #   候选合格信号数门槛
  --param-max-changes 2               #   每个参数候选最多组合修改数
  --max-values-per-param 8            #   每个 tunable 最多测试的候选值
  --evaluation-mode fast              #   候选评估: fast / full
  --full-eval-top 5                   #   fast 模式下完整复评前 N 名
  --reject-overfit                    #   拒绝完整评估中被标记过拟合的候选
  --max-train-val-gap 0.12            #   训练/验证胜率落差上限
  --max-val-test-gap 0.10             #   验证/测试胜率落差上限
  --max-walk-forward-gap 0.12         #   walk-forward 平均训练/测试落差上限
  --min-walk-forward-pass-rate 0.5    #   walk-forward 通过折数比例下限
  --parallel-workers 4                #   并发评估候选，适合大网格/联合搜索
  --joint-top 3                       #   将前 N 个入口/参数候选继续做 exit/risk 联合搜索
  --joint-candidates-per-seed 64      #   每个联合搜索种子的 exit/risk 候选预算
  --fill-policy conservative
  --save-best                         #   将最佳候选保存到策略库
alphaevo evolve <strategy_id>         # 进化策略 (需要 [llm])
  --rounds 3                          #   进化轮数
  --method hybrid                     #   进化方法: llm / param_search / hybrid
  --fill-policy conservative          #   止损/止盈同 K 线冲突策略
  --output reports/                   #   导出演化报告和 research log

# 评估与排行
alphaevo leaderboard                  # 策略排行榜（默认 local scope）
  --scope local|official              #   local 显示本地研究 run；official 只显示 official_v1 通过项
alphaevo compare <id1> <id2>          # 策略对比
alphaevo runs list                    # 列出不可变 run / optimization artifact 索引
alphaevo runs show <run_id>           # 查看 run provenance 与 artifact 路径
alphaevo runs compare <id1> <id2>     # 比较两个 run；不同 protocol 只诊断不排名

# 进化树
alphaevo tree <strategy_id>           # 查看策略进化树
alphaevo tree --all                   # 全局进化树

# 配置
alphaevo config show                  # 查看配置
alphaevo config set <key> <value>     # 设置配置

# 演示
alphaevo demo                         # 快速 demo，默认使用 bundled real-data snapshot
  --synthetic                         #   使用合成数据做开发烟测
  --real                              #   尝试实时数据 demo
alphaevo showcase                     # Star-facing 真实数据 showcase，生成报告
  --live                              #   尝试实时 yfinance，失败回退 snapshot
  --write-docs                        #   更新 docs/reports/showcase_rsi_reversion_real_snapshot.md

# 版本
alphaevo version                      # 显示版本号
```

---

## 13. Async/Sync 设计决策

**决策: 接口定义用 `async def`，MVP 实现中用 `asyncio.run()` 桥接。**

理由：
- DataAdapter 的网络 I/O 天然适合 async（并发获取多只股票数据）
- 回测引擎本身是 CPU-bound，但 per-symbol 循环可以并发
- CLI 层用 Typer（同步），在命令入口处用 `asyncio.run()` 桥接
- 后续如果加 Web API (FastAPI) 可直接复用 async 接口

```python
# CLI 入口桥接模式
@app.command()
def run(strategy_id: str):
    import asyncio
    result = asyncio.run(_run_pipeline(strategy_id))
    _display_result(result)
```

---

## 14. 默认工作流

1. 先判断任务类型：`feat / fix / refactor / docs / test`。
2. 先读现有实现和测试，再动手修改。
3. 只做当前任务直接相关的最小改动。
4. 新增功能必须有对应测试。
5. 改完后执行验证：
   - `python -m pytest tests/unit/`
   - `python -m pytest tests/integration/` (如涉及)
   - `ruff check src/`
   - `mypy src/alphaevo/` (如配置了)

---

## 15. 验证矩阵

| 改动面 | 最低验证 | 完整验证 |
|--------|----------|----------|
| 核心模型 (`models/`) | 单元测试 | 类型检查 + 单元测试 |
| 策略层 (`strategy/`) | DSL 解析测试 + 回测测试 | + 进化闭环集成测试 |
| 回测引擎 (`backtest/`) | 单元测试 + 已知数据验证 | + walk-forward 测试 |
| CLI (`cli/`) | 命令可执行 | + 输出格式快照测试 |
| 数据适配器 (`data/adapters/`) | mock 测试 | + 真实数据冒烟测试 |
| 配置 (`core/config.py`) | 单元测试 | + 环境变量/文件优先级测试 |
| 文档 | 命令/文件名校验 | — |

---

## 16. 文档同步规则

以下文档之间存在交叉引用，修改时必须保持一致：

| 内容 | 权威来源 | 需同步更新 |
|------|----------|------------|
| 综合评分公式 | `AGENTS.md` §10 | `docs/module_tech_specs.md` §6 |
| DSL 规范 | `AGENTS.md` §7 | `docs/technical_design.md` §2.2 |
| CLI 命令列表 | `AGENTS.md` §12 | `src/alphaevo/cli/main.py` |
| 指标清单 | `AGENTS.md` §9 | `docs/module_tech_specs.md` §4 |
| 配置项 | `AGENTS.md` §5 | `src/alphaevo/core/config.py` |
| Canonical run / provenance | `AGENTS.md` §11 | `docs/canonical_evaluation.md`, `docs/capability_roadmap.md`, `src/alphaevo/cli/main.py` |
