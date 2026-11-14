# AlphaEvo 各模块技术方案详细设计

> 本文档为每个模块的技术选型、实现方案、关键决策做详细说明。
> 文档同步规则：DSL 规范以 `AGENTS.md` §7 为准，评分公式以 `AGENTS.md` §10 为准。

### 模块实现状态总览

| 模块 | 设计文档 | 代码实现 | 测试 |
|------|----------|----------|------|
| LLM 客户端 | ✅ §1 | ✅ `core/llm.py` | ✅ |
| 自我进化机制 | ✅ §2 | ✅ `reflection/` + `orchestrator/evolution.py` | ✅ |
| 回测引擎 | ✅ §3 | ✅ `backtest/` | ✅ |
| 组合回测 | ✅ §3 | ✅ `backtest/portfolio.py` | ✅ |
| 指标计算 | ✅ §4 | ✅ `backtest/indicators.py` | ✅ |
| 采样器 | ✅ §5 | ✅ `sampler/` | ✅ |
| 评估器 | ✅ §6 | ✅ `evaluator/` | ✅ |
| Walk-Forward / 稳定性评估 | ✅ §7 | ⚠️ 部分实现于 `evaluator/metrics.py` | ⚠️ |
| 数据适配器 | ✅ §8 | ✅ `data/adapters/` | ✅ |
| 配置管理 | ✅ AGENTS.md §5 | ✅ `core/config.py` | ✅ |
| 策略序列化 | ✅ technical_design §3.0 | ✅ `strategy/dsl/serializer.py` | ✅ |
| Pydantic 模型 | ✅ technical_design §2 | ✅ `models/` | ✅ |
| DSL 解析器 | ✅ AGENTS.md §7 | ✅ `strategy/dsl/parser.py` | ✅ |
| 数据适配器接口 | ✅ technical_design §3.1 | ✅ `data/adapter.py` | ✅ |
| CLI | ✅ AGENTS.md §12 | ✅ `cli/main.py` | ✅ |
| 内置策略模板 | ✅ | ✅ 7 个 YAML 文件 (2个标记 experimental) | ✅ |
| Alpha Factory | ✅ | ✅ `alpha_factory/` (LLM 因子发现 → 进化循环集成) | ✅ |

---

## 一、LLM 客户端：用 LiteLLM 还是自己封装？

### 结论：**用 LiteLLM，但做一层薄封装**

### 分析

| 方案 | 优点 | 缺点 |
|------|------|------|
| 直接用 litellm | 支持 100+ 模型; 自带 Router 多 key 负载均衡; 社区活跃 | API 变化快; 返回格式需标准化; thinking model 需特殊处理 |
| 自己写 HTTP 调用 | 完全可控 | 工作量巨大; 每个 provider 格式不同; 不值得 |
| 用 LangChain | 生态丰富 | 太重; 抽象层过多; 与项目风格不符 |

### 参考 daily_stock_analysis 的做法

DSA 的 `LLMToolAdapter` 是一个 ~400 行的薄封装：
- 输入: `messages` (OpenAI 格式) + `tools` (函数声明)
- 内部: 调 `litellm.completion()` 或 `litellm.Router.completion()`
- 输出: 统一的 `LLMResponse(content, tool_calls, reasoning_content, usage)`
- 亮点: 自动探测 thinking model (deepseek-r1, qwq), 多 model fallback

### AlphaEvo 方案

```python
class LLMClient:
    """AlphaEvo 的 LLM 封装层，比 DSA 更轻更聚焦。"""

    # 核心区别: AlphaEvo 不需要 tool_calling / ReAct loop
    # AlphaEvo 的 LLM 调用场景只有 3 个，全部是 text-in → structured-JSON-out

    def __init__(self, model: str, fallback_models: list[str] = [],
                 api_key: str = None, base_url: str = None):
        # 直接用 litellm.completion()，不需要 Router
        # 除非用户配置了多 key，才启用 Router
        ...

    async def generate_json(self, system_prompt: str, user_prompt: str,
                            response_model: type[BaseModel]) -> BaseModel:
        """核心方法: 输出 Pydantic 模型的结构化 JSON。
        
        内部流程:
        1. 调 litellm.completion(response_format={"type": "json_object"})
        2. 解析返回的 JSON
        3. 尝试 Pydantic 验证
        4. 失败则 json_repair → 重试一次
        """

    async def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        """简单文本生成 (用于 reflection summary 等)。"""
```

### 为什么不需要 Tool Calling / ReAct

DSA 用 tool_calling 是因为它的 Agent 需要动态决定"要不要调 get_realtime_quote"。
AlphaEvo 的 3 个 LLM 调用场景都是确定性的：

| 场景 | 输入 | 输出 | 需要 tool? |
|------|------|------|------------|
| 策略生成 | 自然语言描述 + 指标清单 | Strategy YAML | ❌ 纯文本生成 |
| 失败归因 | 策略 + 评估报告 + 失败案例 | 归因模式 + StrategyChange JSON | ❌ 纯分析 |
| 策略改写 | 当前策略 + 归因 + 修正建议 | 新版策略 YAML | ❌ 纯改写 |

全部是 **structured output** 场景，不需要 ReAct loop。

### 结构化输出的可靠性保障

```python
# 多级 fallback 解析 (参考 DSA 的 parse_dashboard_json)
def _parse_llm_json(raw: str, model_cls: type[BaseModel]) -> BaseModel:
    # 1. 直接 json.loads
    # 2. 提取 ```json ... ``` 代码块
    # 3. 提取最外层 { ... }
    # 4. json_repair 修复常见错误 (尾逗号, 缺引号)
    # 5. Pydantic model_validate
    # 6. 如果全部失败, 重新调一次 LLM 并附带错误提示
```

### 多模型支持策略

```python
# 推荐默认配置
ALPHAEVO_LLM_MODEL=gemini/gemini-3.8-flash        # 默认快速主力模型，适合策略生成与日常反思
ALPHAEVO_LLM_REFLECT_MODEL=                        # 留空则复用上面的
# 如果用户想用更强的模型做深度反思:
ALPHAEVO_LLM_REFLECT_MODEL=anthropic/claude-sonnet-5.5

# 不同场景可配不同模型, 但默认用同一个
```

---

## 二、自我进化机制：到底怎么做？

### 核心理念

**不是让 LLM 凭空写策略，而是让 LLM 做"有数据支撑的反思和微调"。**

进化 ≠ 随机突变。进化 = 归因 + 假设 + 验证。

### 进化的三种方法

#### 方法 A：LLM 反思进化

```
输入:
  - 当前策略 DSL (YAML)
  - 评估报告 (整体指标 + 分环境指标)
  - 10 个最典型失败案例 (含行情数据)
  - 10 个最典型成功案例 (对比用)

LLM 任务:
  1. 分析失败案例的共性模式 (输出 3-5 条)
  2. 对每条模式提出具体修正 (StrategyChange 格式)
  3. 每次最多改 3 个条件 (防止大幅跳变)

输出:
  - failure_patterns: ["追高太晚,放量是出货不是主升", "行业强度不够", ...]
  - proposed_changes: [
      {change_type: "tighten_filter", target: "relative_strength_20d", from: 0.08, to: 0.12},
      {change_type: "add_condition", target: "sector_heat_rank", value: "<=15"},
    ]
  - next_strategy: 完整的新版 Strategy YAML
```

**关键 Prompt 设计：**
```
你是一个量化策略研究员。以下是一个交易策略的回测结果。

## 当前策略
{strategy_yaml}

## 评估结果
- 胜率: {win_rate}%, 盈亏比: {pl_ratio}, 最大回撤: {max_dd}%
- 牛市胜率: {bull_wr}%, 震荡市胜率: {range_wr}%, 熊市胜率: {bear_wr}%

## 10 个典型失败案例
{failure_cases_with_data}

## 10 个典型成功案例
{success_cases_with_data}

请分析失败原因,并给出 **具体的、可执行的** 策略修改建议。
要求:
1. 输出 3-5 条失败模式
2. 每条对应一个 StrategyChange (必须指定具体字段和新值)
3. 最多改 3 个条件
4. 输出完整的新版策略 YAML

以 JSON 格式输出...
```

#### 方法 B：参数网格搜索

```python
class ParamSearchEvolver:
    """对 DSL 中标记为 tunable 的参数做小范围搜索。"""

    def evolve(self, strategy: Strategy, evaluator: Evaluator,
               backtest_engine: BacktestEngine) -> list[Strategy]:
        candidates = []
        for param in strategy.params.tunable:
            # 在 param.range 内按 param.step 枚举
            # target 既可以是 ...value，也可以是
            # entry.triggers[...].indicator / entry.guards[...].indicator
            # entry.conditions[...].indicator / exit.take_profit.target / exit.max_holding_days
            # 或 entry.triggers[...].indicator.fast/.slow
            # 用于指标窗口调参（如 ma60 -> ma55, rsi_14 -> rsi_10,
            # volume_ratio_1d_5d -> volume_ratio_1d_10d,
            # ma5_ge_ma10 -> ma6_ge_ma10）
            for value in arange(param.range[0], param.range[1], param.step):
                # 生成候选策略 (只改这一个参数)
                candidate = self._clone_with_param(strategy, param.target, value)
                candidates.append(candidate)

        # 对每个候选跑快速回测 (可以用较少样本)
        results = [backtest_and_evaluate(c) for c in candidates]

        # 按 confidence_score 排序, 取 top 3
        return sorted(results, key=lambda r: r.score, reverse=True)[:3]
```

**特点**: 不依赖 LLM, 确定性强, 但只能调参数不能加减条件。

#### 方法 C：混合进化 ⭐ 推荐

```
第 1 步: LLM 分析失败案例 → 输出 "大方向" (如: 需要加强行业过滤)
第 2 步: LLM 生成 3 个候选修改方案
第 3 步: 对每个方案, 用参数搜索微调数值参数
第 4 步: 分别回测所有候选
第 5 步: 取 confidence_score 最高的作为下一版
第 6 步: LLM 生成改进总结 (为什么这个版本更好)
```

**这比纯 LLM 稳定得多**：LLM 负责"想方向"，参数搜索负责"找最优值"。

### 进化循环的完整流程

```python
async def evolve(strategy: Strategy, rounds: int = 3,
                 method: EvolutionMethod = "hybrid") -> list[EvolutionNode]:
    nodes = []
    current = strategy

    for round_idx in range(rounds):
        # 1. 采样 + 回测
        batch = await sampler.sample(current)
        backtest_result = await engine.run(current, batch)

        # 2. 评估
        evaluation = evaluator.evaluate(backtest_result, current)

        # 3. 过拟合检查
        if evaluation.anti_overfit.is_overfit:
            # 标记为 pruned, 回退到上一个版本的另一个分支
            nodes.append(EvolutionNode(status="pruned", ...))
            break

        # 4. 反思 + 进化
        if method == "llm":
            next_strategy = await llm_reflect_evolve(current, evaluation)
        elif method == "param_search":
            next_strategy = param_search_evolve(current, evaluation)
        elif method == "hybrid":
            next_strategy = await hybrid_evolve(current, evaluation)

        # 5. 验证新版本
        new_batch = await sampler.sample(next_strategy)
        new_result = await engine.run(next_strategy, new_batch)
        new_eval = evaluator.evaluate(new_result, next_strategy)

        # 6. 是否真的变好了?
        if new_eval.confidence_score > evaluation.confidence_score:
            nodes.append(EvolutionNode(status="active", ...))
            current = next_strategy  # 继续进化
        else:
            nodes.append(EvolutionNode(status="pruned", ...))
            # 可选: 用同一个 evaluation 让 LLM 换个方向再试

    # 7. 选出冠军
    best = max(nodes, key=lambda n: n.evaluation.confidence_score)
    best.status = "champion"
    return nodes
```

### 进化的安全护栏

| 护栏 | 规则 | 目的 |
|------|------|------|
| 最大改动数 | 每轮最多改 3 个条件 | 防止 LLM 大幅跳变 |
| 复杂度上限 | 总条件 > 8 则强制简化 | 防止堆条件过拟合 |
| 必须变好 | confidence_score 不升则 prune | 防止随机游走 |
| 过拟合检测 | train-val gap > 15% 则停止 | 防止训练集过拟合 |
| 最小信号数 | < 30 个信号不算有效 | 防止小样本偏差 |
| 参数敏感度 | 参数扰动 ±10% 性能衰减 > 30% 则警告 | 防止参数脆弱 |

### 进化能力扩展

| 能力 | 实现 | 说明 |
|------|------|------|
| **结构变异 (CHANGE_LOGIC)** | `mutator._change_logic()` | 支持 AND↔OR 逻辑切换，当信号过少且条件过多时自动触发 |
| **因子发现 (DISCOVER_FACTOR)** | `evolution._try_factor_discovery()` → `AlphaFactory.discover()` | 当常规反思无改进方案时，LLM 自动生成新指标代码，经沙盒验证+统计验证后注入策略 |
| **LLM vs 参数搜索基线** | `EvolutionResult.baseline_param_search_score` | 进化完成后自动用纯参数搜索做基线对比，量化 LLM 的增量价值 |
| **组合回测** | `PortfolioBacktester` | 初始资金 $100K，最多 5 个并发持仓，20% 仓位分配，计算组合级收益/回撤/Sharpe |

---

## 三、回测引擎：怎么把 DSL 条件变成真实信号？

### 核心难题

策略 YAML 写了 `indicator: relative_strength_20d, op: ">", value: 0.08`，
但回测引擎需要：
1. 知道 `relative_strength_20d` 怎么算
2. 在每根 K 线上计算它
3. 与 0.08 比较
4. 所有条件都满足才产生买入信号

### 方案：指标注册表 + 条件评估器

```python
# ═══ 指标注册表 ═══

class IndicatorRegistry:
    """注册所有可计算的指标。每个指标是一个纯函数。"""

    _registry: dict[str, IndicatorFn] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(fn: IndicatorFn):
            cls._registry[name] = fn
            return fn
        return decorator

    @classmethod
    def compute(cls, name: str, df: pd.DataFrame, idx: int,
                ctx: "IndicatorContext" = None) -> float | bool:
        """在给定 DataFrame 的第 idx 行计算指标值。ctx 为可选的辅助上下文。"""
        fn = cls._registry[name]
        # 支持两种签名: (df, idx) 和 (df, idx, ctx)
        import inspect
        sig = inspect.signature(fn)
        if len(sig.parameters) >= 3:
            return fn(df, idx, ctx)
        return fn(df, idx)

    @classmethod
    def available(cls) -> list[str]:
        return list(cls._registry.keys())


# ═══ 具体指标实现 ═══

@IndicatorRegistry.register("ma5_above_ma10")
def ma5_above_ma10(df: pd.DataFrame, idx: int) -> bool:
    if idx < 10:
        return False
    ma5 = df["close"].iloc[idx-4:idx+1].mean()
    ma10 = df["close"].iloc[idx-9:idx+1].mean()
    return ma5 > ma10

@IndicatorRegistry.register("relative_strength_20d")
def relative_strength_20d(df: pd.DataFrame, idx: int, ctx: IndicatorContext = None) -> float:
    """个股过去20日涨跌幅 - 基准指数涨跌幅"""
    if idx < 20:
        return 0.0
    stock_return = (df["close"].iloc[idx] / df["close"].iloc[idx-20]) - 1
    benchmark_return = 0.0
    if ctx and ctx.benchmark_df is not None:
        bm = ctx.benchmark_df
        if idx < len(bm) and idx >= 20:
            benchmark_return = (bm["close"].iloc[idx] / bm["close"].iloc[idx-20]) - 1
    return stock_return - benchmark_return

@IndicatorRegistry.register("volume_ratio_1d_5d")
def volume_ratio_1d_5d(df: pd.DataFrame, idx: int, ctx: IndicatorContext = None) -> float:
    if idx < 5:
        return 1.0
    today_vol = df["volume"].iloc[idx]
    avg_5d_vol = df["volume"].iloc[idx-5:idx].mean()  # 过去5天 (不含今天)
    return today_vol / avg_5d_vol if avg_5d_vol > 0 else 1.0

@IndicatorRegistry.register("rsi_14")
def rsi_14(df: pd.DataFrame, idx: int) -> float:
    if idx < 14:
        return 50.0
    changes = df["close"].diff().iloc[idx-13:idx+1]
    gains = changes.clip(lower=0).mean()
    losses = (-changes.clip(upper=0)).mean()
    if losses == 0:
        return 100.0
    rs = gains / losses
    return 100 - (100 / (1 + rs))

# ... 还需实现: close_to_ma10_pct, close_above_ma20,
#     deviation_from_ma20_pct, has_stop_signal, negative_news_score,
#     sector_heat_rank, news_sentiment_score 等
```

### 条件评估器

```python
class ConditionEvaluator:
    """将 StrategyCondition 对象求值为 True/False。支持 AND/OR 逻辑。"""

    OPS = {
        "==": operator.eq,
        "!=": operator.ne,
        ">":  operator.gt,
        ">=": operator.ge,
        "<":  operator.lt,
        "<=": operator.le,
    }

    def evaluate(self, condition: StrategyCondition,
                 df: pd.DataFrame, idx: int, ctx: IndicatorContext = None) -> bool:
        actual = IndicatorRegistry.compute(condition.indicator, df, idx, ctx)
        expected = condition.value
        op_fn = self.OPS[condition.op]
        return op_fn(actual, expected)

    def evaluate_group(self, conditions: list[StrategyCondition],
                       logic: str, df: pd.DataFrame, idx: int,
                       ctx: IndicatorContext = None) -> bool:
        """按 logic (and/or) 组合评估条件组。"""
        results = (self.evaluate(c, df, idx, ctx) for c in conditions)
        if logic == "or":
            return any(results)
        return all(results)  # 默认 AND

    def evaluate_entry(self, entry: StrategyEntry,
                       df: pd.DataFrame, idx: int,
                       ctx: IndicatorContext = None) -> bool:
        """评估完整入场条件: triggers/conditions 触发买入, guards/filters 硬过滤。"""
        triggers = entry.triggers or entry.conditions
        guards = [*entry.guards, *entry.filters]
        trigger_ok = self.evaluate_group(triggers, entry.logic, df, idx, ctx)
        guard_ok = self.evaluate_group(guards, "and", df, idx, ctx)
        return trigger_ok and guard_ok
```

`StrategyEntry.triggers` 表示真正触发买入的信号，`StrategyEntry.guards`
表示市场质量/风控过滤条件。旧版 `conditions` / `filters` 仍兼容：
当 `triggers` 为空时回退到 `conditions`；`guards` 与 `filters` 始终按
AND 过滤，避免“过滤条件本身变成买点”。

### A 股市场规则处理

```python
class MarketRuleChecker:
    """市场特殊规则检查器。架构上兼容多市场, MVP 实现 A 股规则。"""

    def can_buy(self, df: pd.DataFrame, idx: int, rules: MarketRuleConfig) -> bool:
        """检查当日是否可买入。"""
        if not rules:
            return True

        row = df.iloc[idx]

        # 涨跌停检测: 涨停无法买入 (收盘价 == 涨停价)
        if rules.limit_up_down and self._is_limit_up(row):
            return False

        # 停牌检测: 成交量为 0 视为停牌
        if rules.suspension and row.get("volume", 0) == 0:
            return False

        return True

    def can_sell(self, df: pd.DataFrame, idx: int, entry_idx: int,
                 rules: MarketRuleConfig) -> bool:
        """检查当日是否可卖出。"""
        if not rules:
            return True

        row = df.iloc[idx]

        # T+1: 买入当天不可卖出
        if rules.t_plus_1 and idx <= entry_idx:
            return False

        # 跌停无法卖出
        if rules.limit_up_down and self._is_limit_down(row):
            return False

        # 停牌
        if rules.suspension and row.get("volume", 0) == 0:
            return False

        return True

    @staticmethod
    def _is_limit_up(row) -> bool:
        """A 股涨停判断: 涨幅 >= 9.8% (允许误差)"""
        if row.get("prev_close") and row.get("prev_close") > 0:
            change_pct = (row["close"] - row["prev_close"]) / row["prev_close"]
            return change_pct >= 0.098
        return False

    @staticmethod
    def _is_limit_down(row) -> bool:
        """A 股跌停判断: 跌幅 >= 9.8%"""
        if row.get("prev_close") and row.get("prev_close") > 0:
            change_pct = (row["close"] - row["prev_close"]) / row["prev_close"]
            return change_pct <= -0.098
        return False

    @classmethod
    def for_market(cls, market: str) -> "MarketRuleConfig":
        """工厂方法: 返回市场默认规则。"""
        DEFAULTS = {
            "a_share": MarketRuleConfig(t_plus_1=True, limit_up_down=True, suspension=True),
            "hk": MarketRuleConfig(t_plus_1=False, limit_up_down=False, suspension=True),
            "us": MarketRuleConfig(t_plus_1=False, limit_up_down=False, suspension=True),
        }
        return DEFAULTS.get(market, MarketRuleConfig())
```

### 回测引擎核心循环

```python
class BacktestEngine:
    def __init__(self, data_manager: DataManager,
                 condition_evaluator: ConditionEvaluator,
                 market_rule_checker: MarketRuleChecker = None,
                 fill_policy: str = "conservative"):
        self.data = data_manager
        self.evaluator = condition_evaluator
        self.rule_checker = market_rule_checker or MarketRuleChecker()
        self.fill_policy = fill_policy

    async def run(self, strategy: Strategy,
                  batch: SampleBatch) -> BacktestResult:
        signals = []
        # 获取市场规则 (优先用策略自带, 其次按 meta.market 默认)
        market_key = strategy.meta.market.value
        rules = strategy.market_rules.get(
            market_key, MarketRuleChecker.for_market(market_key))

        for symbol in batch.symbols:
            df = await self.data.get_history(
                symbol, batch.date_range[0], batch.date_range[1])

            if df.empty or len(df) < 30:
                continue

            # 构建指标上下文 (基准指数等)
            ctx = await self._build_context(symbol, batch)
            position = None
            entry_idx = None  # 记录买入日索引, 用于 T+1 检查

            for idx in range(30, len(df)):
                # ── 持仓中: 检查退出条件 ──
                if position is not None:
                    # 市场规则: 是否可卖出 (T+1, 跌停, 停牌)
                    if not self.rule_checker.can_sell(df, idx, entry_idx, rules):
                        continue

                    exit_signal = self._check_exit(
                        position, strategy.exit, df, idx, ctx)
                    if exit_signal:
                        signal = self._close_position(position, exit_signal, df, idx)
                        signals.append(signal)
                        position = None
                        entry_idx = None
                    continue  # 持仓中不检查新入场

                # ── 空仓: 检查入场条件 ──
                # 市场规则: 是否可买入 (涨停, 停牌)
                if not self.rule_checker.can_buy(df, idx, rules):
                    continue

                # 使用支持 AND/OR 的 evaluate_entry
                if self.evaluator.evaluate_entry(strategy.entry, df, idx, ctx):
                    position = self._open_position(symbol, df, idx)
                    entry_idx = idx

            # 循环结束仍持仓: 强制平仓
            if position is not None:
                signal = self._force_close(position, df)
                signals.append(signal)

        return BacktestResult(
            strategy_id=strategy.meta.id,
            batch_id=batch.batch_id,
            signals=signals,
            total_signals=len(signals),
            executed_signals=len([s for s in signals if s.exit_price]),
            date_range=batch.date_range,
        )

    def _check_exit(self, position, exit_config, df, idx, ctx=None) -> Optional[ExitSignal]:
        """检查最大持仓、止损、止盈、显式卖出触发器"""
        current_price = df.iloc[idx]["close"]
        holding_days = (df.iloc[idx]["date"] - position.entry_date).days

        # 最大持仓天数
        if holding_days >= exit_config.max_holding_days:
            return ExitSignal(reason=ExitReason.MAX_HOLD, price=current_price)

        # 计算止损风险值 (统一用 risk_amount 表示每股风险)
        risk_amount = self._compute_risk(position, exit_config.stop_loss, df, idx, ctx)

        stop_signal = self._check_stop_loss(position, exit_config.stop_loss, df, idx, ctx)
        take_profit_signal = self._check_take_profit(
            position, exit_config.stop_loss, exit_config.take_profit, df, idx, ctx)

        if stop_signal and take_profit_signal:
            return self._resolve_intrabar_conflict(
                stop_signal, take_profit_signal, position, df, idx)
        if stop_signal:
            return stop_signal
        if take_profit_signal:
            return take_profit_signal

        # 显式卖出触发器：任一满足即按 signal 平仓
        for trigger in exit_config.triggers:
            if condition_evaluator.evaluate_condition(trigger, df, idx, ctx):
                return ExitSignal(reason=ExitReason.SIGNAL, price=current_price)

        return None

    def _resolve_intrabar_conflict(self, stop_signal, take_profit_signal, position, df, idx):
        """同一根 K 线同时触发止损/止盈时，按 fill_policy 选择成交假设。"""
        if self.fill_policy == "optimistic":
            return take_profit_signal
        if self.fill_policy == "close_first":
            return take_profit_signal if df.iloc[idx]["close"] >= position.entry_price else stop_signal
        return stop_signal

    def _compute_risk(self, position, stop_loss_config, df, idx, ctx=None) -> float:
        """统一计算每股风险金额, 供止盈 RR 计算使用。"""
        if stop_loss_config.type == "pct":
            return abs(position.entry_price * stop_loss_config.value)
        elif stop_loss_config.type == "atr":
            atr_name = f"atr_{stop_loss_config.atr_period}" if stop_loss_config.atr_period else "atr"
            atr = IndicatorRegistry.compute(atr_name, df, idx, ctx)
            return atr * (stop_loss_config.multiplier or 2.0)
        else:
            return position.entry_price * 0.05  # 默认 5% 风险
```

> `ATR` 止损支持可选 `atr_period`；省略时回退到 `atr(14)`。

### 对比 DSA 的 BacktestEngine

| 维度 | DSA | AlphaEvo |
|------|-----|------|
| 输入 | AnalysisResult (LLM输出的建议文本) | Strategy DSL (结构化条件) |
| 信号来源 | 从文本推断方向 (`infer_direction_expected`) | 从 DSL 条件精确计算 |
| 评估方式 | 事后验证 (分析已发生, 看后续行情) | 模拟执行 (逐日遍历, 实时判断) |
| 止损止盈 | 从分析报告中提取价格目标 | 从 DSL exit 规则精确执行 |
| 适用场景 | 验证 LLM 分析准确度 | 验证策略 DSL 有效性 |

**结论: 不能直接复用 DSA 的 BacktestEngine，逻辑完全不同。** DSA 是"事后评估LLM预测"，AlphaEvo 是"模拟执行策略DSL"。

### composite 退出条件处理 (v0.3 标准化)

`StopLossConfig.conditions` 从 `list[dict[str, Any]]` 改为 `list[StrategyCondition]`，复用入场条件的统一结构。回测引擎对 composite 退出的处理：任一条件满足即触发止损。

```python
# 旧格式 (废弃):
conditions: [{type: sector_rank_exit, threshold: 10}]

# 新格式 (v0.3):
conditions:
  - indicator: sector_heat_rank
    op: ">"
    value: 10
  - indicator: close_below_ma10
    op: "=="
    value: true
```

### 入场执行方式 (v0.3 新增)

`StrategyEntry.execution` 控制信号触发后的执行时机：

| timing | 说明 | 回测实现 |
|--------|------|----------|
| `next_open` | 次日开盘价买入 | `df.iloc[idx+1]["open"]` (默认) |
| `close` | 当日收盘价买入 | `df.iloc[idx]["close"]` |
| `breakout_high` | 突破当日最高价时买入 | `df.iloc[idx+1]["high"]` (如果突破则用 high 作入场价) |

---

## 四、指标计算：哪些指标要自己实现？

### 策略引用的所有指标清单

从内置策略 YAML 中提取。**实现层级**与 `AGENTS.md` §9 一致：

| 指标名 | 使用策略 | 实现层级 | 数据需求 | MVP 降级 |
|--------|----------|----------|----------|----------|
| `ma5_above_ma10` | 趋势回踩 | **L1 MVP** | OHLCV | — |
| `close_to_ma10_pct` | 趋势回踩 | **L1 MVP** | OHLCV | — |
| `close_above_ma20` | 趋势回踩 | **L1 MVP** | OHLCV | — |
| `volume_ratio_1d_Nd` | 趋势/事件 | **L1 MVP** | OHLCV | — |
| `rsi_N` | 超跌反弹 | **L1 MVP** | OHLCV | — |
| `deviation_from_ma20_pct` | 超跌反弹 | **L1 MVP** | OHLCV | — |
| `has_stop_signal` | 超跌反弹 | **L1 MVP** | OHLCV (K线) | — |
| `volume_shrink_then_rise` | 超跌反弹 | **L1 MVP** | OHLCV | — |
| `ma5_ge_ma10_or_crossing` | 板块轮动 | **L1 MVP** | OHLCV | — |
| `atr / atr_N` | 回测引擎 | **L1 MVP** | OHLCV | — |
| `macd_histogram / macd_histogram_fastN_slowM_signalK` | 趋势/动量过滤 | **L1 MVP** | OHLCV | — |
| `macd_cross_bullish / macd_cross_bullish_fastN_slowM_signalK` | 趋势/动量过滤 | **L1 MVP** | OHLCV | — |
| `bollinger_band_width / bollinger_band_width_Nd / bollinger_band_width_Nd_stdS` | 波动突破 | **L1 MVP** | OHLCV | — |
| `price_above_bollinger_upper / price_above_bollinger_upper_Nd / price_above_bollinger_upper_Nd_stdS` | 波动突破 | **L1 MVP** | OHLCV | — |
| `price_below_bollinger_lower / price_below_bollinger_lower_Nd / price_below_bollinger_lower_Nd_stdS` | 波动突破 | **L1 MVP** | OHLCV | — |
| `breakout_high_Nd` | 突破/新高买点 | **L1 MVP** | OHLCV | — |
| `price_position_Nd` | 区间位置过滤 | **L1 MVP** | OHLCV | — |
| `close_below_ma10` | 退出条件 | **L1 MVP** | OHLCV | — |
| `relative_strength_Nd` | 趋势回踩 | **L2 扩展** | 基准指数 | → 返回 0.0 |
| `st_flag` | 趋势/反转/轮动 | **L2 扩展** | 股票标记 | → 返回 False |
| `sector_heat_rank` | 板块轮动 | **L2 扩展** | 板块数据 | → 返回 1 (通过) |
| `sector_heat_rising_days` | 板块轮动 | **L2 扩展** | 板块时序 | → 返回 99 (通过) |
| `intra_sector_strength_rank_pct` | 板块轮动 | **L2 扩展** | 板块内排名 | → 返回 0.0 (通过) |
| `negative_news_score` | 趋势回踩 | **L3 高级** | 新闻 API | → 价格/量能事件 proxy，缺失时 0.0 |
| `news_sentiment_score` | 事件驱动 | **L3 高级** | 新闻+情感 | → gap + close strength proxy，缺失时 0.5 |
| `days_since_event` | 事件驱动 | **L3 高级** | 事件数据库 | → 最近 proxy event 的天数，缺失时 999 |
| `price_above_pre_event` | 事件驱动 | **L3 高级** | 事件数据库 | → 基于 `pre_event_close` proxy |
| `sector_fund_flow_positive` | 事件驱动 | **L3 高级** | 资金流 | → 返回 True |
| `already_overreacted` | 事件驱动 | **L3 高级** | 复合判断 | → 基于事件锚点涨幅 proxy |
| `sector_risk_flag` | 超跌反弹 | **L3 高级** | 行业数据 | → 返回 False |
| `sector_net_inflow_days` | 板块轮动 | **L3 高级** | 资金流 | → 返回 99 (通过) |

> **MVP 策略可用性**: trend + reversal 策略的核心指标全部在 L1。event + rotation 策略已不再是简单默认放行，但新闻/事件链路仍主要依赖 proxy，结果更适合作为研究参考。详见 `AGENTS.md` §8。

### 分层实现策略

**第 1 层 (MVP, 仅 OHLCV)：窗口模板 + 核心指标** — 可立即实现
```
ma5_above_ma10, close_to_ma10_pct, close_above_ma20,
volume_ratio_1d_Nd, rsi_N, deviation_from_ma20_pct,
has_stop_signal, volume_shrink_then_rise,
ma5_ge_ma10_or_crossing, atr/atr_N,
macd_histogram/macd_histogram_fastN_slowM_signalK,
macd_cross_bullish/macd_cross_bullish_fastN_slowM_signalK,
bollinger_band_width/bollinger_band_width_Nd/bollinger_band_width_Nd_stdS,
price_above_bollinger_upper/price_above_bollinger_upper_Nd/price_above_bollinger_upper_Nd_stdS,
price_below_bollinger_lower/price_below_bollinger_lower_Nd/price_below_bollinger_lower_Nd_stdS,
breakout_high_Nd, price_position_Nd,
momentum_Nd, avg_volume_Nd, days_since_high_Nd, days_since_low_Nd,
rsi_N_zscore, volatility_Nd
```

**第 2 层 (需额外数据源)：5 个指标** — 需要板块/指数数据
```
relative_strength_Nd (需要指数数据)
st_flag (需要股票标记)
sector_heat_rank, sector_heat_rising_days (需要板块数据)
intra_sector_strength_rank_pct (需要板块内排名)
```

**第 3 层 (需要新闻/事件)：6 个指标** — 需要 LLM 或外部 API
```
negative_news_score, news_sentiment_score,
days_since_event, price_above_pre_event,
sector_fund_flow_positive, already_overreacted
```

### 缺失指标的降级 / 代理策略

```python
@IndicatorRegistry.register("negative_news_score")
def negative_news_score(df: pd.DataFrame, idx: int) -> float:
    """优先使用外部新闻数据，缺失时退化为价格/量能事件 proxy。"""
    return price_volume_event_proxy(df, idx).negative_score

@IndicatorRegistry.register("st_flag")
def st_flag(df: pd.DataFrame, idx: int) -> bool:
    """是否ST股。MVP 阶段返回 False，后续从股票信息获取。"""
    # TODO: 从 StockInfo 获取
    return False
```

**原则**: 优先使用真实上下文，其次使用可解释的 proxy；只有两者都不可用时，才回退到中性默认值，确保回测流程可以跑通。

---

## 五、采样器：怎么选股票样本？

### 为什么不能全市场跑？

- A 股 5000+ 只股票 × 2 年日线 ≈ 250 万行数据
- 全跑一次回测可能要 10+ 分钟
- 进化 3 轮 × 每轮跑 2 次 = 60+ 分钟
- 而且全市场跑容易掩盖策略在特定环境的表现

### 采样方案

```python
class AdaptiveSampler:
    """根据策略类型自动选择采样方法。"""

    async def sample(self, strategy: Strategy,
                     date_range: tuple[date, date],
                     n_samples: int = 50) -> SampleBatch:

        # 1. 获取全量股票列表
        all_stocks = await self.data.get_stock_list(strategy.meta.market)

        # 2. 应用 universe 过滤
        filtered = self._apply_universe_filters(all_stocks, strategy.universe)

        # 3. 根据策略类型选择采样方式
        if strategy.meta.category == "trend":
            # 趋势策略: 按近期涨跌幅分层采样
            return self._stratified_by_momentum(filtered, n_samples)
        elif strategy.meta.category == "reversal":
            # 反转策略: 偏向近期超跌的股票
            return self._biased_toward_oversold(filtered, n_samples)
        elif strategy.meta.category == "rotation":
            # 轮动策略: 按行业均匀采样
            return self._stratified_by_sector(filtered, n_samples)
        else:
            # 默认: 随机采样
            return self._random_sample(filtered, n_samples)

    def _stratified_by_momentum(self, stocks, n) -> SampleBatch:
        """分层采样: 强势股 40%, 中性股 30%, 弱势股 30%"""
        # 按近 20 日涨跌幅排序
        # 前 20% = 强势, 中间 60% = 中性, 后 20% = 弱势
        # 从每层按比例抽取
        ...
```

---

## 六、评估器：confidence_score 怎么算？

### 综合评分公式

```python
def compute_confidence_score(metrics: OverallMetrics,
                              anti_fit: AntiFitMetrics,
                              strategy: Strategy) -> float:
    """
    综合评分 = 收益质量 - 风险惩罚 - 过拟合惩罚

    各分项都归一化到 [0, 1]:
    """
    # 收益质量 (越高越好)
    wr_score = min(1.0, metrics.win_rate / 0.7)         # 70%胜率得满分
    ret_score = min(1.0, metrics.avg_return / 0.05)      # 5%平均收益得满分
    pl_score = min(1.0, metrics.profit_loss_ratio / 2.5) # 2.5盈亏比得满分

    # 风险 (越低越好, 反转后越高越好)
    dd_score = max(0, 1.0 - metrics.max_drawdown / 0.30) # 30%回撤得0分
    sharpe_score = min(1.0, max(0, metrics.sharpe_ratio / 2.0))

    # 稳定性 (越高越好)
    consistency = max(0, anti_fit.yearly_consistency)
    sensitivity = max(0, 1.0 - anti_fit.param_sensitivity)

    # 惩罚
    overfit_penalty = 0.0
    if anti_fit.train_val_gap > 0.10:
        overfit_penalty += 0.15
    if anti_fit.val_test_gap > 0.08:
        overfit_penalty += 0.10
    complexity_penalty = strategy.complexity_score * 0.10

    # 加权求和
    score = (
        0.25 * wr_score +
        0.15 * ret_score +
        0.15 * pl_score +
        0.15 * dd_score +
        0.10 * sharpe_score +
        0.10 * consistency +
        0.10 * sensitivity
        - overfit_penalty
        - complexity_penalty
    )

    return max(0.0, min(1.0, score))
```

### 为什么不只看胜率或只看收益？

| 只看胜率的陷阱 | 只看收益的陷阱 |
|----------------|----------------|
| 90% 胜率但盈亏比 0.3 → 总体亏损 | 平均 10% 但胜率 20% → 连续亏损受不了 |
| 高胜率低赔率 → 一次大亏吃掉所有利润 | 高收益高回撤 → 实盘心态崩溃 |

综合评分确保策略在**胜率、收益、风控、稳定性**之间取得平衡。

---

## 七、防过拟合：Walk-Forward 怎么实现？

### 原理

```
传统回测:  [═══════ 全部数据 ═══════]  ← 用同一批数据训练+测试, 必然过拟合

Walk-Forward:
  轮次1: [══ 训练 ══][测试]
  轮次2:    [══ 训练 ══][测试]
  轮次3:       [══ 训练 ══][测试]
  轮次4:          [══ 训练 ══][测试]
                                      ← 每次在"未见数据"上测试
```

### 实现

当前实现已经支持可配置的 walk-forward protocol：

- `backtest.walk_forward_folds`
- `backtest.walk_forward_train_pct`
- `backtest.walk_forward_pass_gap`

当前默认是**按信号时间顺序的滚动折叠**，并在报告中显式输出 protocol 与每折结果；更严格的日历型 `12m → 1m` canonical protocol 仍属于后续增强项。

`alphaevo optimize` 已能把完整评估中的稳定性结果作为候选门槛：
`--reject-overfit` 会拒绝被 train/validation/test 或参数敏感度检查标记为过拟合的候选；
`--max-train-val-gap`、`--max-val-test-gap`、`--max-walk-forward-gap`
限制跨区间表现落差；`--min-walk-forward-pass-rate` 要求足够多滚动折叠通过。
`--objective robust_profit_quality` 则把收益质量与这些稳定性指标合成一个排序分数，
用于优先查看收益与泛化表现更均衡的候选。
在 `--evaluation-mode fast` 下，这些稳健门槛只对 `--full-eval-top`
覆盖到的完整复评候选生效。

```python
class WalkForwardValidator:
    def __init__(self, train_months: int = 12, test_months: int = 1):
        self.train_months = train_months
        self.test_months = test_months

    async def validate(self, strategy: Strategy,
                       total_range: tuple[date, date]) -> WalkForwardResult:
        windows = self._generate_windows(total_range)
        results = []

        for train_start, train_end, test_start, test_end in windows:
            # 1. 在训练期回测 (用于参数优化, 如果 method=param_search)
            train_eval = await self._run_period(strategy, train_start, train_end)

            # 2. 在测试期回测 (评估真实表现)
            test_eval = await self._run_period(strategy, test_start, test_end)

            results.append(WalkForwardWindow(
                train=train_eval, test=test_eval,
                train_range=(train_start, train_end),
                test_range=(test_start, test_end),
            ))

        # 3. 汇总: 所有测试期的平均表现
        avg_test_wr = mean([r.test.overall.win_rate for r in results])
        avg_train_wr = mean([r.train.overall.win_rate for r in results])

        return WalkForwardResult(
            windows=results,
            avg_train_win_rate=avg_train_wr,
            avg_test_win_rate=avg_test_wr,
            train_test_gap=avg_train_wr - avg_test_wr,
            is_robust=abs(avg_train_wr - avg_test_wr) < 0.15,
        )
```

---

## 八、数据适配器：如何兼容独立模式和 DSA 插件模式？

### 架构

```
         ┌─────────────┐
         │ DataManager  │  ← 上层只依赖这个接口
         └──────┬───────┘
                │
    ┌───────────┼───────────┐
    │           │           │
┌───▼───┐ ┌────▼────┐ ┌────▼────┐
│YFinance│ │AKShare  │ │  DSA    │  ← 3 个适配器, 选其一或组合
│Adapter │ │Adapter  │ │Adapter  │
└───────┘ └─────────┘ └─────────┘
```

### 独立模式 (默认)

```python
# YFinance 适配器 — 零配置即可用
class YFinanceAdapter(DataAdapter):
    @property
    def name(self) -> str: return "yfinance"

    async def get_daily_data(self, symbol: str, days: int = 120) -> pd.DataFrame:
        import yfinance as yf
        # 将 A 股代码转换为 yfinance 格式
        yf_symbol = self._convert_symbol(symbol)  # "600519" → "600519.SS"
        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(period=f"{days}d")
        return self._normalize_columns(df)

    def _convert_symbol(self, symbol: str) -> str:
        """统一代码格式转换。"""
        if symbol.startswith("6"):   return f"{symbol}.SS"  # 沪市
        if symbol.startswith("0"):   return f"{symbol}.SZ"  # 深市
        if symbol.startswith("3"):   return f"{symbol}.SZ"  # 创业板
        return symbol  # 美股/港股直接返回
```

### DSA 插件模式

```python
class DSAAdapter(DataAdapter):
    """对接 daily_stock_analysis 的 DataFetcherManager。"""

    def __init__(self, dsa_path: str = None):
        self._dsa_path = dsa_path or self._auto_discover()
        self._fetcher = self._import_fetcher()

    def _auto_discover(self) -> str:
        """自动发现 daily_stock_analysis 安装路径。"""
        candidates = [
            Path.home() / "daily_stock_analysis",
            Path.cwd().parent / "daily_stock_analysis",
        ]
        for p in candidates:
            if (p / "data_provider").is_dir():
                return str(p)
        raise ImportError("daily_stock_analysis not found")

    def _import_fetcher(self):
        """动态导入 DSA 的 DataFetcherManager。"""
        import sys
        sys.path.insert(0, self._dsa_path)
        from data_provider.base import DataFetcherManager
        return DataFetcherManager()

    async def get_daily_data(self, symbol: str, days: int = 120) -> pd.DataFrame:
        # 调用 DSA 的 6 源 fallback 能力
        df, source = self._fetcher.get_daily_data(symbol, days)
        return df
```

### 自动选择

```python
def create_data_manager(config: dict) -> DataManager:
    adapter_name = config.get("AlphaEvo_DATA_ADAPTER", "yfinance")

    if adapter_name == "dsa":
        return DataManager([DSAAdapter(config.get("DSA_PATH"))])
    elif adapter_name == "akshare":
        return DataManager([AKShareAdapter(), YFinanceAdapter()])  # akshare 优先, yfinance fallback
    else:
        return DataManager([YFinanceAdapter()])
```

---

## 九、SQLite 存储：表结构和 Store 模式

### 为什么用 SQLite 而不是 ORM？

| 方案 | 优点 | 缺点 |
|------|------|------|
| SQLAlchemy ORM | 类型安全, 迁移工具 | 重, 对这个项目来说过度设计 |
| 原生 sqlite3 | 轻, 零依赖, 快 | SQL 字符串, 无迁移 |
| **sqlite3 + 轻量 Store 模式** | 平衡 | — |

### Store 模式

```python
class BaseStore:
    """轻量 Store 基类, 封装 sqlite3 连接。"""

    def __init__(self, db_path: str = "~/.alphaevo/alphaevo.db"):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._init_tables()

    @abstractmethod
    def _init_tables(self): ...


class StrategyStore(BaseStore):
    def _init_tables(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS strategies (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                version INTEGER NOT NULL,
                parent_id TEXT,
                category TEXT NOT NULL,
                market TEXT NOT NULL,
                dsl_yaml TEXT NOT NULL,
                description TEXT NOT NULL,
                complexity_score REAL,
                status TEXT DEFAULT 'active',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

    def save(self, strategy: Strategy) -> None:
        yaml_str = yaml.dump(strategy.model_dump(), allow_unicode=True)
        self._conn.execute(
            "INSERT OR REPLACE INTO strategies (...) VALUES (...)",
            (strategy.meta.id, ..., yaml_str, ...)
        )
        self._conn.commit()

    def load(self, strategy_id: str) -> Strategy:
        row = self._conn.execute(
            "SELECT dsl_yaml FROM strategies WHERE id = ?",
            (strategy_id,)
        ).fetchone()
        return StrategyParser().parse_yaml(row["dsl_yaml"])

    def get_evolution_tree(self, root_id: str) -> list[EvolutionNode]:
        """递归查询进化树。"""
        ...
```

---

## 十、模块间数据流总览

```
用户输入 "强趋势回踩放量反包"
        │
        ▼
┌─ LLM Client ─┐
│ 自然语言 → YAML │ ← generate_json(prompt, StrategyGenerateResponse)
└───────┬───────┘
        │ Strategy 对象
        ▼
┌─ Sampler ─────┐
│ Universe过滤   │ ← DataManager.get_stock_list()
│ 分层采样 50只  │
└───────┬───────┘
        │ SampleBatch
        ▼
┌─ BacktestEngine ──────────────────────┐
│ for symbol in batch:                   │
│   df = DataManager.get_history(symbol) │
│   for each day:                        │
│     IndicatorRegistry.compute(...)     │
│     ConditionEvaluator.evaluate(...)   │
│     → TradeSignal                      │
└───────┬───────────────────────────────┘
        │ BacktestResult (signals[])
        ▼
┌─ Evaluator ───┐
│ 计算指标       │ → OverallMetrics
│ 分环境统计     │ → RegimeMetrics[]
│ 防过拟合检查   │ → AntiFitMetrics
│ 留一环境诊断   │ → RegimeHoldoutMetrics
│ 高压窗口评测   │ → StressWindowMetrics
│ 综合评分       │ → confidence_score
└───────┬───────┘
        │ EvaluationReport
        ▼
┌─ Reflection (LLM) ──────────────────┐
│ 输入: 策略 + 评估 + 失败案例         │
│ 分析: 失败模式归因                   │
│ 输出: StrategyChange[] + 新策略      │
└───────┬─────────────────────────────┘
        │ ReflectionResult + 新 Strategy
        ▼
    回到 Sampler (下一轮进化)
        │
        ▼ (N轮后)
┌─ Leaderboard ─┐
│ 按 score 排序  │
│ 选出 champion  │
│ 渲染报告       │
└───────────────┘
```

---

## 十一、各模块实现优先级和依赖关系图

```
                    ┌──────────────┐
                    │ p1a-indicators│ ← 零依赖, 最先开工
                    └──┬───┬───┬───┘
                       │   │   │
              ┌────────┘   │   └────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │p1a-yfin  │ │p1a-mock  │ │p1b-cond  │
        └────┬─────┘ └────┬─────┘ └────┬─────┘
             │            │            │
             ▼            │            ▼
        ┌──────────┐      │      ┌──────────┐
        │p1c-samplr│      │      │p1b-engine│ ← 核心模块
        └────┬─────┘      │      └──┬───┬───┘
             │            │         │   │
             │            ▼         │   ▼
             │      ┌──────────┐    │ ┌──────────┐
             │      │p1b-tests │    │ │p1e-db    │
             │      └──────────┘    │ └────┬─────┘
             │                      ▼      │
             │               ┌──────────┐  │
             │               │p1c-eval  │  │
             │               └──┬───┬───┘  │
             │                  │   │      │
             │                  │   ▼      │
             │                  │ ┌────────┤
             │                  │ │p1c-anti│
             │                  │ └────────┘
             │                  ▼
             │            ┌──────────┐
             └───────────►│p1e-orch  │◄── p1d-mutator
                          └────┬─────┘
                               ▼
                         ┌──────────┐
                         │p1e-cli   │
                         └────┬─────┘
                               ▼
                         ┌──────────┐
                         │p1f-integ │
                         └──────────┘

并行线 (LLM, 与上面无依赖):
  p1d-llm → p1d-gen + p1d-reflect → p1d-mutator → 汇入 p1e-orch
```
