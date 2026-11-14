<div align="center">

<img src="../sources/alphaevo_terminal_v1_agent_evolution.svg" alt="AlphaEvo Logo" width="660">

# 🧬 AlphaEvo

**开源「RSI (Evo) + Stock」递归自我进化股票研究引擎**

*策略进化 (L1) × 因子发现 (L2) × 研究机制自举 (L3) —— 用反幻觉零假设控制与双时态前瞻影子账本，把 Autoresearch 安全落到股票投研*

[![GitHub stars](https://img.shields.io/github/stars/ZhuLinsen/alphaevo?style=social)](https://github.com/ZhuLinsen/alphaevo/stargazers)
[![CI](https://github.com/ZhuLinsen/alphaevo/actions/workflows/ci.yml/badge.svg)](https://github.com/ZhuLinsen/alphaevo/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](../LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://github.com/ZhuLinsen/alphaevo)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](../CONTRIBUTING.md)

[English](../README.md) | [中文](README_CN.md) | [RSI + Stock 架构详述](rsi_stock_architecture.md) | [完整技术设计](technical_design.md)

</div>

---

## ⚡ 为什么「RSI (Evo) + Stock」需要双时钟架构

> **核心不变量：历史回测只负责否决（Veto），样本外前瞻影子账本才决定晋升（Promote）。**

通用代码/ML 领域的 `autoresearch` 放在股票里极易崩溃，因为个股信号信噪比低、同日同板块聚类导致有效样本数 $N_{\text{eff}} \ll N$、多轮试错推高虚假 Sharpe，且自动生成因子容易混入未来信息。**AlphaEvo** 把递归自我进化做成可证伪的股票投研闭环：

| 维度 | 通用 RSI (`autoresearch` / `OpenEvolve`) | 传统股票 LLM Agent | **AlphaEvo (`RSI (Evo) + Stock`)** |
|---|---|---|---|
| **自我改进闭环** | 针对低噪声代码/Loss 单层棘轮 | 单次提示词选股或静态回测 | **L1 策略 + L2 因子 + L3 研究机制三层进化** + **MAP-Elites** |
| **多重检验防刷榜** | 通常单验证集 | 很少统计失败试次 | 只增不减 **`TrialLedger`** + **Deflated Sharpe Ratio (DSR)** |
| **零边际反幻觉校验** | 无 | 无 | **双层 Null-Signal Control**（随机洗牌市场 **0 误晋升**） |
| **未来函数防御** | 不涉及时序截断 | 依赖口头约束 | **运行时 PIT 未来噪声探针** + 严格 `<= as_of` 内存切片 |
| **样本外晋升机制** | 直接提交代码 | 缺乏前瞻跟踪 | **SHA-256 哈希链影子账本**、$N_{\text{eff}}$ 去相关、瞬间熔断 `HALTED` |

---

## 🚀 30 秒快速开始（无需 API Key）

```bash
git clone https://github.com/ZhuLinsen/alphaevo.git && cd alphaevo
pip install -e .
alphaevo showcase                                      # 用内置真实行情快照跑一轮完整策略进化
alphaevo rsi run --program examples/rsi_stock_program.md  # 运行 L0 守护的双时钟 RSI 棘轮与 results.tsv
```

<p align="center">
  <img src="demo.gif" alt="AlphaEvo Demo" width="700">
</p>

### 按场景选入口

| 目标场景 | 命令 | 数据源 / LLM 需求 |
|---|---|---|
| **30 秒真实快照 Showcase** | `alphaevo showcase` | 内置 yfinance 快照 / 无需 Key |
| **RSI (Evo) + Stock 自主进化 (`program.md`)** | `alphaevo rsi run --program examples/rsi_stock_program.md` | 快照或 `--live` / LLM 可选 |
| **Tier-2 洗牌行情零晋升审计** | `alphaevo rsi null-audit --worlds 3 --rounds 3` | 块洗牌零信号世界 / 否 |
| **前瞻影子账本全链路** | `alphaevo shadow register\|emit\|backfill\|status\|verify` | 严格 PIT 行情 / 否 |
| **异构多模型对抗议会（强制跨家族）** | `alphaevo rsi run --method llm --council-proposer codex:... --council-prosecutor agy:...` | 任意 / `proposer != prosecutor` |
| **美股 / A 股实盘 LLM 与参数进化** | `pip install -e ".[llm,data-full]"` → `alphaevo evolve rsi_reversion_v1 --method llm` | `yfinance` / `akshare` / 是 |

*美股场景可选配置 `ADANOS_API_KEY` 接入 Reddit/X 情绪事件增强；未配置时自动降级为价量事件代理指标。*

---

## 🏗️ 双时钟系统架构与股票 DSL

```
快时钟（离线假设搜索 —— L1 / L2 / L3 RSI）
  program.md (RSICharter) ─▶ BlindSpotMiner（诊断 L1 策略 vs L2 因子 + 运行时 PIT 探针）
           ▲                          │
           │ Keep / Revert            ▼
  MAP-Elites 档案库 ◀── HeterogeneousCouncil（提案家族 != 审计家族 + Executable Pinned Scars）
  (市场环境×风格×周期)                │
                                      ▼
                    L0 不可变反幻觉真相核 (`alphaevo/truth`):
                    TrialLedger · Deflated Sharpe (DSR) · NullSignalController
                                      │ 仅放行通过否决门限的候选
                                      ▼
慢时钟（样本外前瞻问责 —— `alphaevo/forward`）
  CandidateSpec ─▶ commit_signals(as_of <= t) ─▶ 回填 T+5/10/20 扣费收益 ─▶ N_eff + 非对称状态机
                   [SHA-256 哈希链 + 行业硬上限]                            (DRAFT→SHADOW→CANARY→ACTIVE|HALTED)
```

策略与交易剧本（`SetupSpec`）均以可读 YAML DSL 声明（完整规范见 [RSI+Stock 架构文档](rsi_stock_architecture.md) 与 [AGENTS.md](../AGENTS.md)）：

```yaml
meta:
  id: trend_pullback_rsi_v1
  market: a_share
  category: reversal
setup:
  trigger_bar: "rsi_14 < 30 and volume_ratio_1d_5d >= 1.5"
  invalidation_rule: "close < ma60 * 0.96"
  target_rr: 3.0
  min_rr: 2.0
entry:
  logic: or
  conditions:
    - { indicator: rsi_14, op: "<", value: 30 }
    - { indicator: close_to_ma60_pct, op: "<=", value: 0.06 }
exit:
  stop_loss: { type: pct, value: 0.08 }
  take_profit: { type: rr, value: 4.0 }
  max_holding_days: 14
params:
  tunable:
    - { target: entry.conditions[indicator=rsi_14].value, range: [22, 36], step: 2 }
```

---

## 🔬 真实验证与诚实熔断

AlphaEvo 同时公开有效进化案例与「被防过拟合/零对照诚实拦下」的案例：

| 验证场景 | 数据与协议 | 实测结果 | 详细报告 |
|---|---|---|---|
| **零 Key 真实快照 Showcase** | `rsi_reversion_v1 -> v7`，美股科技股历史快照 | `0 -> 37` 信号，胜率 `48.6%`，均益 `+2.94%`，最大回撤 `23.6%`，`8.1% -> 68.7%` | [快照报告](reports/showcase_rsi_reversion_real_snapshot.md) |
| **实时行情 LLM 进化** | `rsi_reversion_v1 -> v3`，实时 `yfinance` + `--method llm` | `0 -> 498` 信号，胜率 `52.6%`，均益 `+1.22%`，`8.1% -> 56.3%` | [LLM 验证报告](reports/2026-04-10-real-llm-validation_CN.md) |
| **防过拟合诚实拦截** | `ma_crossover_v1` / `sector_rotation_leader_v1` | 候选虽有局部提升，因 `train_val_gap=18.9%` 等未过泛化检验而拒绝晋升 | [LLM 验证报告](reports/2026-04-10-real-llm-validation_CN.md) |
| **L0 Tier-2 零对照审计** | `alphaevo rsi null-audit`，块洗牌纯噪声收益率世界 | **误晋升数 = 0** | [架构详述](rsi_stock_architecture.md) |
| **L2 因子发现沙盒** | `alphaevo factor discover AAPL` | 提出 `3` 个因子 → 沙盒通过 `3` 个 → 验证并注册 `2` 个（IR `1.76` / `1.50`） | [因子实践文档](cookbook/07_factor_discovery_walkthrough.md) |

---

## 📊 CLI 核心命令

```bash
# 双时钟 RSI (Evo) + Stock
alphaevo rsi run --program examples/rsi_stock_program.md   # L1/L2 演化棘轮并导出 results.tsv
alphaevo rsi null-audit --worlds 3                         # 验证随机洗牌行情下 0 误晋升
alphaevo rsi mechanism-ab --rounds 3                       # L3 元机制样本外校准 AB 对比
alphaevo shadow register <id> && alphaevo shadow emit <id> # 前瞻影子账本 SHA-256 哈希链提交
alphaevo shadow backfill && alphaevo shadow status         # T+5/10/20 收益衰减与 N_eff 诊断
alphaevo truth trials && alphaevo truth pit-probe f.py     # L0 试验账本 DSR 与未来噪声探针

# 策略研究、参数优化与经典 LLM 进化
alphaevo run <id> && alphaevo evolve <id> --method llm     # 单策略全回测 / LLM 自我进化
alphaevo optimize <id> --objective robust_profit_quality   # 买卖点联合优化与过拟合门控
alphaevo factor discover <symbol> && alphaevo leaderboard  # LLM 因子发现与策略排行榜
```

完整子命令与参数说明请运行 `alphaevo --help` 或 `alphaevo <cmd> --help`。

- **多数据源与 DSA 插件**：原生支持 `yfinance`（美股/港股）与 `akshare`（A 股），也可通过 `DSAAdapter` 对接关联项目 [daily_stock_analysis](https://github.com/ZhuLinsen/daily_stock_analysis)。
- **Trajectory 数据飞轮**：每次进化均可导出 `trajectory.jsonl`、`sharegpt.jsonl`（SFT）与 `preference.jsonl`（DPO）——见 [Trajectory Data Flywheel](trajectory_data_flywheel.md)。
- **多维防过拟合**：训练/验证/测试时间严格隔离、滚动 Walk-Forward、CPCV、DSR 多重检验惩罚与参数 ±10% 敏感度检验。

---

## 🗺️ 路线图

- [x] **研究底座与 DSL**：YAML DSL + `SetupSpec`、信号级回测器、多维评估器与成熟度检查
- [x] **双时钟 RSI (Evo) + Stock**：`TrialLedger` + DSR、Null Control、运行时 PIT 探针、Co-STEER L2、可执行伤疤、异构对抗议会与双时态影子账本
- [ ] **组合级多策略资金分配**：跨策略协同组合回测与组合风险预算控制
- [ ] **Web Evolution Lab 工作台**：可视化进化树、影子账本监控面板与 MAP-Elites 热力图

---

## 🤝 参与贡献、免责声明与许可证

欢迎提 PR / Issue，详见 [CONTRIBUTING.md](../CONTRIBUTING.md)；采用 **Apache-2.0** 协议开源（[LICENSE](../LICENSE)）。

> **免责声明**：**本项目仅供科研与学习使用，不构成任何投资建议。** 回测与前瞻模拟结果不代表未来实盘收益，使用本软件造成的任何盈亏均由使用者自行承担。

<table>
  <tr>
    <td align="center" width="180">
      <a href="https://www.xiaohongshu.com/user/profile/61594417000000000201fa68" target="_blank">
        <img src="../sources/xiaohongshu.png" width="130" alt="小红书"><br>
        <b>小红书 📱</b>
      </a>
    </td>
    <td valign="top" style="padding-left: 20px">
      <b>📬 联系与关联项目</b><br><br>
      🐛 <a href="https://github.com/ZhuLinsen/alphaevo/issues">提交 GitHub Issue</a> — Bug 反馈 / 功能建议<br>
      📧 <a href="mailto:zhuls345@gmail.com">zhuls345@gmail.com</a> — 商务与学术交流<br>
      🔗 <a href="https://github.com/ZhuLinsen/daily_stock_analysis">daily_stock_analysis</a> — AI 驱动每日股票智能分析系统
    </td>
  </tr>
</table>
