# Self-Evolving Stock Agent (AlphaEvo) — 技术设计文档

> **文档同步规则**: 本文档与 `AGENTS.md` 交叉引用。DSL 规范以 `AGENTS.md` §7 为准，评分公式以 `AGENTS.md` §10 为准。修改时必须同步。

## 一、系统架构总览

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CLI / API 入口                              │
│                    (Typer + Rich / FastAPI)                          │
│               CLI 是同步的, 通过 asyncio.run() 桥接                   │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────────┐
│                     Orchestrator 编排层                              │
│  strategy_generate → sample_select → backtest_run → metrics_eval   │
│  → failure_analyze → strategy_refine → re_test → leaderboard       │
└──┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┘
   │          │          │          │          │          │
┌──▼───┐ ┌───▼───┐ ┌────▼────┐ ┌───▼───┐ ┌───▼────┐ ┌──▼────────┐
│Data  │ │Strat- │ │Sampler  │ │Back-  │ │Evalua- │ │Reflection │
│Layer │ │egy    │ │Layer    │ │test   │ │tor     │ │Layer      │
│      │ │Layer  │ │         │ │Engine │ │Layer   │ │           │
└──┬───┘ └───┬───┘ └────┬────┘ └───┬───┘ └───┬────┘ └──┬────────┘
   │         │          │          │          │         │
┌──▼─────────▼──────────▼──────────▼──────────▼─────────▼─────────┐
│                     Models / Schemas (Pydantic v2)                │
└─────────────────────────────────────────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────────┐
│                     Infrastructure                                  │
│  ┌──────────┐ ┌────────────┐ ┌──────────┐ ┌───────────────────┐    │
│  │ LiteLLM  │ │ SQLite DB  │ │ YAML I/O │ │ Data Adapters     │    │
│  │ (可选)   │ │ (storage)  │ │ (DSL)    │ │ (yfin/ak/dsa)    │    │
│  └──────────┘ └────────────┘ └──────────┘ └───────────────────┘    │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ AppConfig (Pydantic Settings, 环境变量 + YAML + 默认值)       │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### Async/Sync 桥接策略

- 所有业务层接口定义为 `async def`（DataAdapter、BacktestEngine、Sampler 等）
- CLI 层用 Typer（同步），在命令入口处用 `asyncio.run()` 桥接
- DataAdapter 的网络 I/O 天然适合 async（并发获取多只股票）
- 回测引擎本身是 CPU-bound，但 per-symbol 可以并发
- 后续如加 Web API (FastAPI) 可直接复用 async 接口

### 1.1 Web 端兼容性设计

当前仓库仍以 CLI 为主入口，但后续 Web 化遵循以下约束：

1. 核心研究逻辑继续停留在 `orchestrator/`, `backtest/`, `evaluator/`, `reflection/`。
2. Web/API 层不直接复用 CLI 输出字符串，而是消费稳定的 Pydantic 契约。
3. 预留 `src/alphaevo/web/` 作为 **契约层**，只放 request/response DTO、manifest、view models，不强依赖 FastAPI。
4. Web 端建议采用“异步作业 + 进度流 + 工件输出”的工作流，而不是同步阻塞式调用。

建议的未来入口结构：

```text
src/alphaevo/web/
  __init__.py
  contracts.py        # 稳定 DTO / manifest / 视图模型

future optional:
src/alphaevo/web_api/
  app.py              # FastAPI app
  routes/*.py         # HTTP / SSE / WS endpoints
  dependencies.py     # auth / store / pipeline wiring
```

这样可以保证：

- CLI 与 Web 共用同一套核心 pipeline
- Web 端不反向污染核心域模型
- FastAPI 仍可保持 optional extra，不进入 core dependency

---

## 二、核心数据模型

### 2.1 MarketSnapshot — 市场快照

```python
class MarketSnapshot(BaseModel):
    symbol: str                    # "000001.SZ"
    name: str                      # "平安银行"
    date: date
    market: MarketType             # a_share / hk / us

    price: PriceData               # open, high, low, close, prev_close
    volume: VolumeData             # volume, amount, turnover_rate
    indicators: TechnicalIndicators # MA, MACD, RSI, BOLL, KDJ, etc.
    fundamentals: Optional[FundamentalData]  # PE, PB, ROE, etc.
    news: list[NewsItem]           # 近期新闻
    sector: Optional[str]          # 所属行业
    market_context: MarketContext   # 大盘状态、市场情绪
```

### 2.2 Strategy — 策略定义

```python
class StrategyMeta(BaseModel):
    id: str                        # "trend_pullback_rebound_v1"
    name: str                      # "强趋势回踩放量反包"
    version: int                   # 1
    parent_id: Optional[str]       # 进化树父节点
    created_at: datetime           # UTC 时间
    market: MarketType
    category: StrategyCategory     # trend / reversal / event / rotation / framework
    tags: list[str]
    status: StrategyStatus         # active / pruned / champion / draft
    preferred_regime: list[MarketRegime] = []  # v0.3: 适用市场环境

    @computed_field
    def family_id(self) -> str:    # 从 id 中提取策略家族名
    @computed_field
    def complexity_score(self) -> float:  # 自动计算的复杂度 (sigmoid)

class StrategyCondition(BaseModel):
    indicator: str
    op: Literal["==", "!=", ">", ">=", "<", "<="]
    value: float | bool | str

class EntryExecution(BaseModel):   # v0.3 新增
    timing: Literal["next_open", "close", "breakout_high"] = "next_open"
    slippage: float = 0.001        # 默认从 AppConfig 读取

class StrategyEntry(BaseModel):
    logic: Literal["and", "or"] = "and"  # 条件组合逻辑
    triggers: list[StrategyCondition] = []  # v0.5: 真正触发买入的信号
    guards: list[StrategyCondition] = []    # v0.5: 硬过滤条件，始终 AND
    conditions: list[StrategyCondition] = []  # 兼容旧版触发条件
    filters: list[StrategyCondition] = []     # 兼容旧版过滤条件，始终 AND
    execution: EntryExecution = EntryExecution()  # v0.3 新增

class StopLossConfig(BaseModel):
    type: str = "pct"              # pct, atr, price_level, pct_from_low, composite
    value: Optional[float] = None
    multiplier: Optional[float] = None
    atr_period: Optional[int] = None  # 仅 ATR 止损使用，默认 14
    reference: Optional[str] = None
    conditions: Optional[list[StrategyCondition]] = None  # v0.3: composite 标准化

class StrategyExit(BaseModel):
    triggers: list[StrategyCondition] = []  # v0.4: 显式卖出触发器，任一满足即 signal 退出
    stop_loss: StopLossConfig
    take_profit: TakeProfitConfig
    max_holding_days: int = 10

class TunableParam(BaseModel):
    target: str                    # "entry.triggers[indicator=rsi_14].value"
                                 # or "entry.guards[indicator=relative_strength_20d].value"
                                 # or "entry.conditions[indicator=rsi_14].value" (legacy)
                                 # or "entry.conditions[indicator=close_above_ma60].indicator"
                                 # or "entry.conditions[indicator=rsi_14].indicator"
                                 # or "entry.conditions[indicator=volume_ratio_1d_5d].indicator"
                                 # or "entry.conditions[indicator=relative_strength_20d].indicator"
                                 # or "entry.conditions[indicator=ma5_ge_ma10_or_crossing].indicator.fast"
                                 # or "entry.conditions[indicator=ma5_ge_ma10_or_crossing].indicator.slow"
                                 # or "entry.conditions[indicator=macd_histogram].indicator.fast"
                                 # or "entry.conditions[indicator=macd_histogram].indicator.slow"
                                 # or "entry.conditions[indicator=macd_histogram].indicator.signal"
                                 # or "entry.conditions[indicator=bollinger_band_width].indicator"
                                 # or "entry.conditions[indicator=bollinger_band_width].indicator.std"
                                 # or "entry.triggers[indicator=breakout_high_20d].indicator"
                                 # or "entry.guards[indicator=price_position_120d].indicator"
                                 # or "exit.take_profit.target" / "exit.stop_loss.atr_period"
                                 # or "exit.max_holding_days"
                                 #    (indicator / MA / MACD / Bollinger / ATR / holding tuning)
    range: tuple[float, float]
    step: float
    label: Optional[str] = None

class Strategy(BaseModel):
    meta: StrategyMeta
    description: str               # 人类可读描述
    universe: UniverseConfig       # 选股范围
    entry: StrategyEntry           # 入场条件
    exit: StrategyExit             # 出场规则
    params: StrategyParams         # 可调参数定义
    market_rules: dict[str, MarketRuleConfig] = {}  # 市场规则
```

> `ATR` 止损支持可选 `atr_period`；未显式设置时默认使用 `14`。

> **v0.3 变更**: composite 退出的 `conditions` 从 `list[dict]` 改为 `list[StrategyCondition]`；
> 新增 `EntryExecution`（入场时机+滑点）；新增 `StrategyMeta.preferred_regime`。

> **v0.4 变更**: 新增 `StrategyExit.triggers`。它与 `entry.conditions` 共用
> `StrategyCondition` 结构，但在持仓期间按 OR 语义执行；命中后以当前 close
> 生成 `ExitReason.SIGNAL`。止损/止盈仍优先于显式卖出触发器。

> **v0.5 变更**: 新增 `StrategyEntry.triggers` / `StrategyEntry.guards`。
> `triggers` 表示真正买点；`guards` 表示硬过滤。旧版
> `conditions` / `filters` 仍兼容。新增 `BacktestConfig.fill_policy`，
> 用于处理同一根 K 线同时触发止损和止盈的成交歧义。

### 2.3 SampleBatch — 采样批次

```python
class SampleBatch(BaseModel):
    batch_id: str
    strategy_id: str
    symbols: list[str]
    date_range: tuple[date, date]
    market_regimes: list[MarketRegime]
    sampling_method: SamplingMethod  # representative / regime_based / strategy_scoped
    sampling_reason: str
```

### 2.4 BacktestResult — 回测结果

```python
class TradeSignal(BaseModel):
    symbol: str
    signal_date: date
    direction: Literal["long", "short", "skip"]
    entry_price: float
    exit_price: Optional[float]
    exit_date: Optional[date]
    exit_reason: Literal["stop_loss", "take_profit", "max_hold", "signal"]
    return_pct: float
    holding_days: int

class BacktestResult(BaseModel):
    strategy_id: str
    batch_id: str
    signals: list[TradeSignal]
    total_signals: int
    executed_signals: int
    skipped_signals: int
    date_range: tuple[date, date]
```

### 2.5 EvaluationReport — 评估报告

```python
class OverallMetrics(BaseModel):
    win_rate: float
    avg_return: float
    profit_loss_ratio: float
    max_drawdown: float
    sharpe_ratio: float
    signal_count: int
    avg_holding_days: float
    max_consecutive_loss: int

class RegimeMetrics(BaseModel):
    regime: MarketRegime
    win_rate: float
    avg_return: float
    signal_count: int

class EvaluationReport(BaseModel):
    strategy_id: str
    batch_id: str
    overall: OverallMetrics
    by_regime: list[RegimeMetrics]
    by_sector: dict[str, OverallMetrics]
    failure_cases: list[TradeSignal]    # 最典型失败案例
    top_patterns: list[str]             # LLM 识别的失败模式
    confidence_score: float             # 综合评分
    anti_overfit: AntiFitMetrics        # 防过拟合指标
    regime_holdout: RegimeHoldoutMetrics | None
    stress_windows: StressWindowMetrics | None
```

### 2.6 ReflectionResult — 反思结果

```python
class StrategyChange(BaseModel):
    change_type: Literal["tighten_filter", "loosen_filter", "add_condition",
                         "remove_condition", "adjust_exit", "change_universe"]
    target: str                    # 修改目标字段路径
    from_value: Any
    to_value: Any
    reason: str                    # LLM 归因理由

class ReflectionResult(BaseModel):
    strategy_id: str
    evaluation_id: str
    failure_patterns: list[str]    # 失败归因聚类
    proposed_changes: list[StrategyChange]
    next_strategy_id: str          # 新版策略ID
    next_strategy: Strategy        # 完整新策略
    reflection_summary: str        # LLM 反思总结
```

### 2.7 EvolutionNode — 进化树节点

```python
class EvolutionNode(BaseModel):
    strategy_id: str
    parent_id: Optional[str]
    version: int
    changes_from_parent: list[StrategyChange]
    evaluation: EvaluationReport
    status: Literal["active", "pruned", "champion"]
    created_at: datetime
```

---

### 2.8 AppConfig — 统一配置

```python
class LLMConfig(BaseModel):
    model: str = "gemini/gemini-3.8-flash"
    reflect_model: Optional[str] = None  # 留空则复用 model
    api_key: Optional[str] = None        # 仅通过环境变量
    base_url: Optional[str] = None

class DataConfig(BaseModel):
    adapter: str = "auto"                # auto / tencent / yfinance / akshare / dsa
    cache_dir: Path = Path.home() / ".alphaevo" / "cache"
    dsa_path: Optional[str] = None        # daily_stock_analysis bridge
    source_failure_threshold: int = 3     # repeated failures before cooldown
    source_cooldown_seconds: float = 60.0

class BacktestConfig(BaseModel):
    slippage: float = 0.001
    commission: float = 0.0003

class EvolutionConfig(BaseModel):
    max_rounds: int = 5
    max_changes_per_round: int = 3

class AppConfig(BaseModel):
    """统一配置，优先级: CLI参数 > 环境变量 > 项目配置 > 用户配置 > 默认值"""
    llm: LLMConfig = LLMConfig()
    data: DataConfig = DataConfig()
    backtest: BacktestConfig = BacktestConfig()
    evolution: EvolutionConfig = EvolutionConfig()
    db_path: Path = Path.home() / ".alphaevo" / "alphaevo.db"
```

---

## 三、模块接口设计

### 3.0 Config & Serialization

```python
class ConfigManager:
    """配置加载器"""
    def load(self, cli_overrides: dict = {}) -> AppConfig:
        """加载配置: 合并 CLI > 环境变量 > 项目文件 > 用户文件 > 默认值"""

    def save_user_config(self, config: AppConfig) -> None:
        """保存到 ~/.alphaevo/config.yaml"""

    def init_interactive(self) -> AppConfig:
        """交互式初始化配置 (alphaevo init)"""


class StrategySerializer:
    """Strategy → YAML 序列化器 (与 StrategyParser 互为逆操作)"""

    def to_yaml(self, strategy: Strategy) -> str:
        """序列化为 YAML 字符串"""

    def to_file(self, strategy: Strategy, path: Path) -> None:
        """序列化并写入文件"""

    def to_dict(self, strategy: Strategy) -> dict:
        """转为可 YAML 序列化的 dict (处理枚举/日期等)"""
```

### 3.1 Data Layer

```python
class DataAdapter(ABC):
    """数据源适配器抽象基类"""
    @abstractmethod
    async def get_daily_data(self, symbol: str, days: int = 120) -> pd.DataFrame: ...

    @abstractmethod
    async def get_realtime_quote(self, symbol: str) -> Optional[RealTimeQuote]: ...

    @abstractmethod
    async def get_stock_list(self, market: MarketType) -> list[StockInfo]: ...

    @abstractmethod
    async def get_sector_data(self, symbol: str) -> Optional[SectorInfo]: ...

class DataManager:
    """统一数据管理器，支持多数据源 fallback"""
    def __init__(self, adapters: list[DataAdapter], *, failure_threshold: int = 3,
                 cooldown_seconds: float = 60.0): ...
    async def get_snapshot(self, symbol: str, date: date) -> MarketSnapshot: ...
    async def get_history(self, symbol: str, start: date, end: date) -> pd.DataFrame: ...
    async def get_realtime_quote(self, symbol: str) -> Optional[RealTimeQuote]: ...
    def health_status(self) -> list[DataSourceHealth]: ...

class DataSourceHealth(BaseModel):
    """数据源健康状态：success/failure/consecutive_failures/disabled/last_error/recent_errors"""
```

默认 `auto` 链路为：`dsa`（配置了 `dsa_path` 才启用）→ `tencent` → `akshare` → `yfinance`。
其中 `tencent` 是直接 HTTP K 线/实时行情源，承担 A 股 OHLCV 稳定性；DSA/AkShare/YFinance 继续提供股票池与丰富上下文 fallback。

### 3.2 Strategy Layer

```python
class StrategyParser:
    """解析 YAML DSL 为 Strategy 对象 [已实现]"""
    def parse_file(self, path: Path) -> Strategy: ...
    def parse_yaml(self, content: str) -> Strategy: ...
    def parse_directory(self, directory: Path) -> list[Strategy]: ...
    def validate(self, strategy: Strategy) -> list[str]: ...

class StrategySerializer:
    """Strategy 对象序列化为 YAML [已实现]"""
    def to_yaml(self, strategy: Strategy) -> str: ...
    def to_file(self, strategy: Strategy, path: Path) -> None: ...
    def to_dict(self, strategy: Strategy) -> dict: ...

class StrategyGenerator:
    """LLM 驱动的策略生成 [已实现, 需要 llm optional]"""
    async def from_description(self, description: str) -> Strategy: ...
    async def from_template(self, template: str, params: dict) -> Strategy: ...

class StrategyStore:
    """策略持久化存储 (SQLite + YAML 文件) [已实现]"""
    def save(self, strategy: Strategy) -> None: ...
    def load(self, strategy_id: str) -> Strategy: ...
    def list_all(self) -> list[StrategyMeta]: ...
    def list_builtin(self) -> list[Strategy]: ...  # 从 strategies/builtin/ 加载
    def get_evolution_tree(self, root_id: str) -> list[EvolutionNode]: ...
```

### 3.3 Sampler Layer

```python
class SamplingStrategy(ABC):
    """采样策略抽象基类"""
    @abstractmethod
    async def sample(self, strategy: Strategy, date_range: tuple[date, date],
                     n_samples: int = 50) -> SampleBatch: ...

class RepresentativeSampler(SamplingStrategy):
    """代表性市场采样：各风格各抽一些"""

class RegimeBasedSampler(SamplingStrategy):
    """按市场环境采样：上涨/震荡/下跌/极端"""

class StrategyScopedSampler(SamplingStrategy):
    """按策略适用范围采样"""

class AdaptiveSampler:
    """自适应采样管理器，根据策略类型选择采样方法"""
    def select_sampler(self, strategy: Strategy) -> SamplingStrategy: ...
    async def sample(self, strategy: Strategy, ...) -> SampleBatch: ...
```

### 3.4 Backtest Engine

```python
class BacktestEngine:
    """回测引擎"""
    async def run(self, strategy: Strategy, batch: SampleBatch,
                  data_manager: DataManager) -> BacktestResult: ...

    def apply_entry_conditions(self, snapshot: MarketSnapshot,
                                strategy: Strategy) -> bool: ...

    def apply_exit_conditions(self, position: Position,
                               snapshot: MarketSnapshot,
                               strategy: Strategy) -> Optional[ExitSignal]: ...
```

### 3.5 Evaluator Layer

```python
class Evaluator:
    """多维度评估器"""
    def evaluate(self, backtest_result: BacktestResult,
                 strategy: Strategy) -> EvaluationReport: ...

    def compute_metrics(self, signals: list[TradeSignal]) -> OverallMetrics: ...
    def compute_regime_metrics(self, signals: list[TradeSignal],
                                regimes: dict) -> list[RegimeMetrics]: ...
    def compute_confidence_score(self, metrics: OverallMetrics,
                                  anti_fit: AntiFitMetrics) -> float: ...

class WalkForwardValidator:
    """Walk-Forward 滚动验证（当前为可配置 rolling folds，后续可扩展为日历型 12m→1m 协议）"""
    async def validate(self, strategy: Strategy,
                       train_months: int = 12,
                       test_months: int = 1,
                       total_range: tuple[date, date] = ...) -> list[EvaluationReport]: ...
```

### 3.6 Reflection Layer

```python
class FailureAnalyzer:
    """失败归因分析"""
    async def analyze(self, evaluation: EvaluationReport,
                      failure_cases: list[TradeSignal],
                      strategy: Strategy) -> list[str]: ...

class StrategyMutator:
    """策略变异器"""
    async def llm_reflect(self, strategy: Strategy,
                           evaluation: EvaluationReport,
                           failures: list[str]) -> list[StrategyChange]: ...

    def param_search(self, strategy: Strategy,
                     evaluation: EvaluationReport,
                     n_candidates: int = 10) -> list[Strategy]: ...

    async def hybrid_evolve(self, strategy: Strategy,
                             evaluation: EvaluationReport) -> Strategy: ...
```

### 3.7 Orchestrator

```python
class EvolutionPipeline:
    """端到端进化流水线"""

    async def run_single(self, strategy: Strategy) -> EvaluationReport:
        """单次运行：采样 → 回测 → 评估"""

    async def evolve(self, strategy: Strategy,
                     rounds: int = 3,
                     method: EvolutionMethod = "hybrid") -> list[EvolutionNode]:
        """多轮进化：运行 → 反思 → 改写 → 再运行"""

    async def run_full_pipeline(self, description: str) -> EvolutionTree:
        """完整流水线：描述 → 生成策略 → 多轮进化 → 输出排行"""
```

---

## 四、数据库 Schema

> 状态: **未实现**。下文为目标 schema 设计。

```sql
-- 策略版本表
CREATE TABLE strategies (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    version INTEGER NOT NULL,
    parent_id TEXT REFERENCES strategies(id),
    category TEXT NOT NULL,
    market TEXT NOT NULL,
    dsl_yaml TEXT NOT NULL,         -- 完整策略 YAML
    description TEXT NOT NULL,
    complexity_score REAL,
    status TEXT DEFAULT 'active',   -- active / pruned / champion
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 采样批次表
CREATE TABLE sample_batches (
    id TEXT PRIMARY KEY,
    strategy_id TEXT REFERENCES strategies(id),
    symbols TEXT NOT NULL,          -- JSON array
    date_start DATE NOT NULL,
    date_end DATE NOT NULL,
    method TEXT NOT NULL,
    reason TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 回测结果表
CREATE TABLE backtest_results (
    id TEXT PRIMARY KEY,
    strategy_id TEXT REFERENCES strategies(id),
    batch_id TEXT REFERENCES sample_batches(id),
    signals_json TEXT NOT NULL,     -- JSON array of TradeSignal
    total_signals INTEGER,
    executed_signals INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 评估报告表
CREATE TABLE evaluations (
    id TEXT PRIMARY KEY,
    strategy_id TEXT REFERENCES strategies(id),
    backtest_id TEXT REFERENCES backtest_results(id),
    win_rate REAL,
    avg_return REAL,
    profit_loss_ratio REAL,
    max_drawdown REAL,
    sharpe_ratio REAL,
    signal_count INTEGER,
    confidence_score REAL,
    full_report_json TEXT,          -- 完整 EvaluationReport JSON
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 反思记录表
CREATE TABLE reflections (
    id TEXT PRIMARY KEY,
    strategy_id TEXT REFERENCES strategies(id),
    evaluation_id TEXT REFERENCES evaluations(id),
    failure_patterns TEXT,          -- JSON array
    changes_json TEXT,              -- JSON array of StrategyChange
    next_strategy_id TEXT REFERENCES strategies(id),
    summary TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 排行榜表
CREATE TABLE leaderboard (
    strategy_id TEXT PRIMARY KEY REFERENCES strategies(id),
    rank INTEGER,
    composite_score REAL,
    win_rate REAL,
    avg_return REAL,
    max_drawdown REAL,
    stability_score REAL,
    last_evaluated TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## 五、LLM Prompt 设计要点

> LLM 功能全部依赖 `alphaevo[llm]` 可选依赖。无 LLM 时 `alphaevo run` 仍可工作（用已有策略回测），仅 `alphaevo evolve` 和 `alphaevo strategy create` 需要 LLM。

### 5.1 策略生成 Prompt
- 输入：用户自然语言描述 + 可用指标列表（`IndicatorRegistry.available()`）
- 输出：完整 YAML DSL（通过 `StrategySerializer.to_yaml()` 验证格式）
- 约束：必须包含 entry/exit/universe，参数必须在合理范围内
- 实现位置：`src/alphaevo/strategy/generator.py`

### 5.2 失败归因 Prompt
- 输入：策略 DSL + 评估报告 + 10个典型失败案例的详细数据
- 输出：3-5 条失败模式 + 每条对应的修正建议
- 约束：修正建议必须是可执行的 StrategyChange 结构
- 实现位置：`src/alphaevo/reflection/analyzer.py`

### 5.3 策略改写 Prompt
- 输入：当前策略 + 失败归因 + 修正建议
- 输出：新版策略 YAML
- 约束：每次最多改3个条件（由 `EvolutionConfig.max_changes_per_round` 控制），避免大幅跳变
- 实现位置：`src/alphaevo/reflection/mutator.py`

### 5.4 市场环境识别 Prompt (Phase 3)
- 输入：近 60 日大盘数据 + 板块轮动数据 + 市场情绪指标
- 输出：MarketRegime 分类 + 置信度 + 推荐策略类型

---

## 六、防过拟合框架

```python
class AntiFitMetrics(BaseModel):
    """防过拟合指标"""
    train_win_rate: float
    val_win_rate: float
    test_win_rate: float
    train_val_gap: float           # 训练-验证胜率差距
    val_test_gap: float            # 验证-测试胜率差距
    yearly_consistency: float      # 年度一致性 (1 - std/mean)
    param_sensitivity: float       # 参数扰动后性能衰减
    complexity_penalty: float      # 条件数量惩罚

    @property
    def is_overfit(self) -> bool:
        return (self.train_val_gap > 0.15 or
                self.val_test_gap > 0.10 or
                self.param_sensitivity > 0.3)
```

---

## 七、兼容性适配层

> 所有具体适配器放在 `src/alphaevo/data/adapters/`。

### 7.1 独立数据适配器

```python
# src/alphaevo/data/adapters/yfinance.py [已实现]
class YFinanceAdapter(DataAdapter):
    """yfinance 数据源，支持美股/港股/A股(有限)"""

# src/alphaevo/data/adapters/akshare.py [已实现]
class AkShareAdapter(DataAdapter):
    """akshare 数据源，支持 A 股全量"""

# src/alphaevo/data/adapters/tencent.py [已实现]
class TencentAshareAdapter(DataAdapter):
    """腾讯直接 K 线/实时行情源，作为 A 股 OHLCV 稳定性优先 fallback"""
```

### 7.2 DSA 插件适配器

```python
# src/alphaevo/data/adapters/dsa.py [已实现]
class DSAAdapter(DataAdapter):
    """对接 daily_stock_analysis 的 DataFetcherManager"""

    def __init__(self, dsa_path: str = None):
        # 自动发现 daily_stock_analysis 安装路径
        # 导入其 DataFetcherManager
        ...
```

### 7.3 策略导入

```python
class DSAStrategyImporter:
    """从 daily_stock_analysis 导入自然语言策略 [待实现, 需要 LLM]"""

    async def import_from_yaml(self, yaml_path: Path) -> Strategy:
        # 读取 DSA 格式的策略 YAML
        # LLM 补全 DSL 字段 (entry/exit/universe/params)
        ...
```
