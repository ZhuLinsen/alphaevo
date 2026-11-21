# Canonical Evaluation / Run Provenance

更新时间: 2026-11-21

> **实现状态**: 核心不可变 Run Artifact 契约（`manifest.json`、`strategy_snapshot.yaml`、`config_snapshot.json`、`data_fingerprint.json`、`evaluation.json`、`trades.jsonl`、`report.md`、`run_artifacts` SQLite 索引、`alphaevo runs list/show/compare`、`alphaevo leaderboard --scope local|official` 及 `official_v1` 降级门控）已在 `src/alphaevo/provenance/` 与 `src/alphaevo/leaderboard/` 落地；完整 `alphaevo data bundle create` 不可变数据包阶段作为下一步扩展。

## 1. 目标

AlphaEvo 需要把 `run` / `evolve` / `optimize` 产出的研究结果逐步变成可追踪、可复现、可审计的不可变工件。

当前增长优先级下，完整 artifact / official leaderboard 不是 Star-facing Showcase Release 的主角。Showcase 阶段只实现最小 provenance 背书；本文档定义的是后续系统化落地的目标契约。

这不是为了做实盘交易执行，也不是为了做大而全金融终端。AlphaEvo 的核心边界仍然是：

> 策略研究引擎：策略假设 -> 回测验证 -> 失败归因 -> 结构化改写 -> 再验证 -> 经验沉淀。

Canonical evaluation 要解决的问题：

- 同一次策略研究到底用了什么策略版本、代码版本、数据、配置和评估协议。
- 结果能不能 replay，还是依赖可变数据源重新抓取。
- 本地研究结果和 official leaderboard 结果是否能比较。
- optimize 是否只展示 best，还是保留完整搜索证据链。

## 2. 核心原则

1. Artifact 文件是唯一真源，SQLite 只做索引。
2. 每次运行追加新工件，不覆盖旧结果。
3. 完整 artifact 阶段不新增重依赖，默认使用 JSON / JSONL / YAML / Markdown。
4. Parquet / pyarrow 只能作为 optional extra，不进入 core dependency。
5. Dirty code 可以跑本地研究，但不能进入 official leaderboard。
6. Mutable provider 可以跑本地研究，但 official 评测未来必须绑定不可变 data bundle。
7. 不同 protocol 的 run 可以诊断性 compare，但不能直接排名。

## 3. Run ID

推荐格式：

```text
YYYYMMDDTHHMMSSZ_<strategy_id>_<hash8>
```

示例：

```text
20260429T143012Z_ma_crossover_v1_a1b2c3d4
```

`hash8` 至少由以下内容派生：

- strategy YAML snapshot
- code version / git commit
- data adapter
- date range
- evaluation protocol
- 结果相关 config snapshot

如果是 optimize，使用 `optimization_session_id`，规则同上。

## 4. Artifact 目录

普通 run：

```text
reports/runs/<run_id>/
  manifest.json
  strategy_snapshot.yaml
  config_snapshot.json
  data_fingerprint.json
  evaluation.json
  trades.jsonl
  report.md
```

evolve：

```text
reports/evolutions/<run_id>/
  manifest.json
  initial_strategy_snapshot.yaml
  final_strategy_snapshot.yaml
  config_snapshot.json
  data_fingerprint.json
  rounds.jsonl
  evaluation.json
  trajectory/
  report.md
```

optimize：

```text
reports/optimizations/<optimization_session_id>/
  manifest.json
  strategy_snapshot.yaml
  config_snapshot.json
  data_fingerprint.json
  search_space.json
  candidates.jsonl
  best_candidate.json
  saved_strategy.yaml
  report.md
```

## 5. Manifest Contract

`manifest.json` 最少包含：

```json
{
  "run_id": "20260429T143012Z_ma_crossover_v1_a1b2c3d4",
  "kind": "run",
  "created_at": "2026-04-29T14:30:12Z",
  "requested_protocol": "fast_research_v1",
  "effective_protocol": "fast_research_v1",
  "strategy": {
    "id": "ma_crossover_v1",
    "version": 1,
    "snapshot_hash": "sha256:..."
  },
  "code_version": {
    "source": "git",
    "commit": "...",
    "dirty": false,
    "diff_hash": null
  },
  "data": {
    "adapter": "yfinance",
    "date_range": {"start": "2024-01-01", "end": "2026-04-29"},
    "fingerprint_hash": "sha256:...",
    "reproducibility": "refetch_dependent",
    "data_mutability": "mutable_provider",
    "data_bundle_id": null
  },
  "config": {
    "snapshot_hash": "sha256:..."
  },
  "determinism": {
    "random_seed": 12345,
    "policy": ["provider_mutable"]
  },
  "artifacts": {
    "artifact_dir": "reports/runs/20260429T143012Z_ma_crossover_v1_a1b2c3d4",
    "report": "report.md",
    "evaluation": "evaluation.json",
    "trades": "trades.jsonl"
  },
  "summary": {
    "score": 0.56,
    "win_rate": 0.52,
    "signal_count": 498,
    "max_drawdown": 0.18
  },
  "status": "completed"
}
```

## 6. Config Snapshot

`config_snapshot.json` 使用白名单，不全量保存环境变量或完整配置文件。

完整 artifact 阶段记录结果相关字段：

- data adapter
- date range
- sampling 参数
- backtest slippage / commission / fill policy
- evaluation protocol
- LLM provider / model / temperature
- optimize objective / gates / seed / worker count
- evolve rounds / method / mutation limits

完整 artifact 阶段不记录：

- API key
- token
- database URL
- proxy credential
- 任意无关环境变量

## 7. Data Fingerprint

普通本地 run 至少记录：

- symbol
- row count
- start date / end date
- OHLCV 字段 hash
- adapter name
- adapter version（能取到时）
- cache hit / miss

`data.reproducibility` 取值：

- `replayable`: 绑定不可变 data bundle 或本地 snapshot。
- `refetch_dependent`: 需要重新从 provider 抓取，结果可能漂移。
- `non_reproducible`: 数据来源、窗口或关键字段不完整。

完整 artifact 阶段不实现完整 data bundle；后续 data bundle 阶段再做：

- `alphaevo data bundle create`
- `data_bundle_id`
- fixed universe
- normalized OHLCV snapshot
- bundle hash
- official rerun

## 8. Evaluation Protocols

### `fast_research_v1`

用于本地快速迭代。

允许：

- dirty code
- mutable provider
- 较小样本
- refetch-dependent data fingerprint

### `full_research_v1`

用于完整本地研究。

要求：

- train / validation / test 指标
- walk-forward
- stress-window
- data fingerprint
- 明确 anti-overfit diagnostics

### `official_v1`

用于官方复评和 official leaderboard。

完整 artifact 阶段只实现 gate，不实现完整 data bundle。

要求：

- clean git worktree
- 固定 data bundle（P1 完整支持）
- 固定市场、时间窗口、手续费、滑点、fill policy
- 最小信号数
- anti-overfit gates

完整 artifact 阶段中如果用户请求 `official_v1` 但条件不满足，系统必须拒绝或明确降级，并记录：

- `requested_protocol`
- `effective_protocol`
- `protocol_downgrade_reason`

## 9. Determinism Policy

每个 run / evolve / optimize 都必须记录 `random_seed`。

采样器、候选生成、参数搜索应从 session seed 派生子 seed。

`determinism.policy` 可组合：

- `deterministic`: 同环境可复现。
- `llm_nondeterministic`: 涉及 LLM 输出。
- `provider_mutable`: 数据源可变。
- `dirty_code`: worktree 未提交。
- `parallel_order_nondeterministic`: 并行候选评估可能影响同分排序。

涉及 LLM 时，manifest 记录：

- provider
- model
- temperature
- prompt hash
- response hash
- call count
- token / cost（能取到时）

完整 prompt / response 默认不进入 manifest。需要训练数据或审计时，通过 trajectory artifact 或显式 `--save-llm-trace` 控制。

## 10. SQLite 索引

完整 artifact 阶段只新增最小索引表，不把数据库变成 artifact 真源。

建议表：`run_artifacts`

字段：

- `run_id`
- `kind`: `run` / `evolve` / `optimize`
- `strategy_id`
- `strategy_version`
- `protocol_name`
- `created_at`
- `artifact_dir`
- `score`
- `win_rate`
- `signal_count`
- `code_dirty`
- `data_reproducibility`
- `status`

完整 manifest / trades / evaluation 仍在 artifact 文件中。

## 11. CLI

完整 artifact 阶段新增：

```bash
alphaevo runs list
alphaevo runs show <run_id>
alphaevo runs compare <run_id1> <run_id2>
```

`runs compare` 在完整 artifact 阶段只做高信号 diff：

- score / win_rate / avg_return / max_drawdown / signal_count
- protocol
- code dirty
- data reproducibility
- strategy snapshot hash 是否变化
- config hash 是否变化
- data fingerprint hash 是否变化
- 简化 strategy DSL diff 路径（如果有 YAML snapshot）

不同 protocol 的 compare 必须标记：

```text
comparable: false
```

这种 compare 只用于诊断差异，不给 winner。

## 12. Leaderboard

`leaderboard` 必须拆 scope：

```bash
alphaevo leaderboard --scope local
alphaevo leaderboard --scope official
```

规则：

- `local`: 本地所有完成 run 可见，但必须显示 protocol、dirty、data reproducibility。
- `official`: 只显示 `official_v1` gate 通过的 run。
- 不同 protocol 不混排成一个 winner。
- 如果没有 official run，显示暂无 official run，不用本地 run 冒充。

## 13. Report Provenance

Markdown report 必须包含简洁 `Run Provenance` 区块：

- run id
- protocol
- strategy snapshot hash
- code commit
- code dirty
- data reproducibility
- data adapter
- date range
- config hash
- artifact dir

如果 `code_dirty=true` 或 `data_reproducibility != replayable`，报告必须提示：这是本地研究结果，不是 official benchmark。

## 14. 非目标

完整 artifact 阶段不做：

- 实盘交易执行
- 公共 Strategy Hub
- 完整 data bundle
- Web 工作台
- 图表型 run compare
- trade-level attribution compare
- 多租户 / RBAC / 审计日志

这些能力可以在 canonical provenance 稳定后进入后续阶段。
