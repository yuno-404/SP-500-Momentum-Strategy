# Trade-Level Strategy Analysis Report

> S&P 500 Sector Momentum Strategy — Fixed Capital, No Compounding
>
> Generated: 2026-03-08 | Backtest Period: 20 Years (~2006-03 to 2026-03)

---

## Executive Summary

本報告以 **固定資本 $30,000** 對每個月度交易周期獨立評估策略效果，不計入複利。目的是隔離策略本身的選股與進出場品質，排除資金成長帶來的績效膨脹。

**最終結果：4/6 項指標通過 — 策略具潛力，但進出場時機與預測穩定性需改善。**

| 指標 | 結果 | 門檻 | 判定 |
|------|------|------|------|
| Mathematical Expectancy | $83.64/trade | > 0 | PASS |
| \|IC\| (Information Coefficient) | 0.0605 | > 0.05 | PASS |
| ICIR (IC Information Ratio) | 0.1509 | > 0.5 | FAIL |
| Profit Factor | 1.79 | > 1.5 | PASS |
| Win Rate | 56.7% | > 45% | PASS |
| Total Trade Efficiency | 13.5% | > 30% | FAIL |

---

## 1. Methodology

### 1.1 No-Compounding Design

傳統回測使用 `portfolio_value *= (1 + net_return)` 進行複利累積。本分析刻意移除複利：

- 每月以固定 $30,000 資本重新配置
- 每檔股票依 sector AUM 權重分配資金：`allocated = capital × weight`
- 買入整數股：`shares = int(allocated / entry_price)`
- 獨立計算每筆交易的 P&L、效率、MFE/MAE

### 1.2 Transaction Costs

- 交易成本（單邊）：0.1%
- 滑價成本（單邊）：0.05%
- 每筆來回總成本：`actual_invested × 0.3%`
- 20 年總交易成本：**$20,696.09**

### 1.3 Signal Lag

使用 T+1 信號延遲（`signal_lag_months=1`）：本月的持倉根據上月底的動量信號決定，模擬實際執行中無法即時交易的情境。

### 1.4 Warm-up Period

前 7 個月不交易，確保 26 週（126 交易日）動量資料完整可用。

---

## 2. Core Strategy Metrics

| Metric | Value |
|--------|-------|
| Fixed Capital per Period | $30,000 |
| Total Trades (stock-months) | 2,091 |
| Total Months Traded | 233 |
| Avg Stocks per Month | 9.0 |
| Win Rate | 56.7% |
| Loss Rate | 43.3% |
| Avg Win (per stock-trade) | $335.22 (+10.12%) |
| Avg Loss (per stock-trade) | $246.06 (-7.00%) |
| **Expectancy per trade** | **$83.64** |
| Profit Factor | 1.79 |
| Total Net P&L | $174,883.88 |
| Avg Monthly Net P&L | $750.57 |
| Total Transaction Costs | $20,696.09 |

### Mathematical Expectancy

```
Expectancy = (Win Rate × Avg Win) - (Loss Rate × Avg Loss)
           = (0.567 × $335.22) - (0.433 × $246.06)
           = $190.07 - $106.34
           = $83.64 per trade
```

**Verdict: VIABLE** — 每筆交易平均正期望值 $83.64。

---

## 3. Information Coefficient (IC) & ICIR

| Metric | Value |
|--------|-------|
| IC Observations (months) | 233 |
| Mean IC | 0.0605 |
| Std IC | 0.4013 |
| ICIR (Mean IC / Std IC) | 0.1509 |

### Interpretation

- **IC (Spearman Rank Correlation)**：每月動量分數 vs 實際報酬的排序相關性
  - `|IC| > 0.05` → **PASS**：動量因子具有統計顯著的預測能力
- **ICIR (IC Information Ratio)**：衡量 IC 的穩定性
  - `ICIR > 0.5` → **FAIL (0.1509)**：預測力不穩定，月度 IC 波動極大（Std = 0.4013）

**Alpha Factor 評估：Significant but UNSTABLE**

動量信號平均而言有效，但在某些月份預測力極差甚至反向，降低了策略的可靠性。

---

## 4. Trade Efficiency

| Metric | Value | Meaning |
|--------|-------|---------|
| Avg Entry Efficiency | 57.2% | 進場點距離月內最低點的距離比 |
| Avg Exit Efficiency | 56.1% | 出場點距離月內最高點的距離比 |
| Avg Total Efficiency | 13.5% | 捕獲月內價格區間的比例 |

### Formulas

```
Entry Efficiency = (Highest - Entry) / (Highest - Lowest)
                 → 100% = 買在最低點, 0% = 買在最高點

Exit Efficiency  = (Exit - Lowest) / (Highest - Lowest)
                 → 100% = 賣在最高點, 0% = 賣在最低點

Total Efficiency = (Exit - Entry) / (Highest - Lowest)
                 → 捕獲了月內多少幅度的獲利
```

### Analysis

- Entry/Exit 各約 57%/56%，屬於中等水準
- **Total Efficiency 僅 13.5%（FAIL）**：
  - 雖然進出場各自不差，但合計捕獲的價格區間很小
  - 說明月度持有期間內股價波動大，但策略只捕獲了一小部分
  - 改善方向：考慮更精確的出場機制（trailing stop profit），或縮短持有期

---

## 5. MFE / MAE Analysis

| Metric | Value |
|--------|-------|
| Avg MFE (max unrealized gain) | +8.78% |
| Avg MAE (max unrealized loss) | -5.77% |
| Avg MFE Captured | -191.3% |
| Avg MAE on winning trades | -2.27% |
| Losing trades once profitable | 66% (avg MFE: +4.88%) |

### Key Findings

1. **MFE Captured 為負 (-191.3%)**：
   - 平均每筆交易在持有期間曾達到 +8.78% 的未實現獲利
   - 但最終實際報酬遠低於 MFE，表示策略在最佳出場點後持續持有，利潤大幅回吐
   - 負值代表許多交易從盈轉虧

2. **Winning trades MAE = -2.27%**：
   - 獲利交易在過程中平均最大回撤僅 -2.27%，進場時機相對精準

3. **66% 的虧損交易曾經獲利（avg MFE: +4.88%）**：
   - 超過三分之二的虧損交易在持有期間曾有平均 +4.88% 的浮盈
   - 若能在適當時機止盈，大量虧損交易本可轉為獲利
   - **這是策略最大的改善空間**

---

## 6. Monthly P&L Summary

| Metric | Value |
|--------|-------|
| Profitable Months | 147 / 233 (63%) |
| Avg Profitable Month | $2,012 |
| Avg Losing Month | -$1,405 |
| Best Month | $10,441 |
| Worst Month | -$6,123 |
| Median Month | $669 |
| Std Dev Monthly P&L | $2,278 |

---

## 7. Sector Performance Breakdown

| Sector | Trades | Win Rate | Avg Return | Total P&L | Avg MFE | Avg MAE |
|--------|--------|----------|------------|-----------|---------|---------|
| Information Technology | 233 | 58% | +4.74% | $46,145 | +13.05% | -7.39% |
| Financials | 233 | 55% | +2.50% | $40,264 | +8.70% | -5.66% |
| Health Care | 233 | 54% | +2.70% | $27,902 | +9.14% | -5.77% |
| Consumer Discretionary | 227 | 57% | +4.30% | $20,020 | +11.52% | -6.79% |
| Industrials | 233 | 61% | +2.45% | $15,966 | +8.69% | -6.92% |
| Consumer Staples | 233 | 55% | +1.62% | $10,902 | +5.62% | -3.88% |
| Energy | 233 | 56% | +1.89% | $7,389 | +8.16% | -5.60% |
| Materials | 233 | 58% | +2.94% | $3,794 | +8.70% | -5.40% |
| Utilities | 233 | 58% | +1.31% | $2,501 | +5.53% | -4.51% |

### Sector Insights

- **IT 與 Financials 貢獻最大**：合計佔總 P&L 的 49%（$86,409 / $174,884）
- **Industrials 勝率最高 (61%)**：穩定但單筆獲利幅度較小
- **Consumer Discretionary 平均報酬最高 (+4.30%)**：但 MFE 也高（+11.52%），代表利潤回吐嚴重
- **Utilities 和 Materials 貢獻較弱**：可能不適合動量策略的板塊

---

## 8. Recent Performance (Last 20 Months)

以下為 2024-08 至 2026-03 的月度 P&L 概覽：

| Month | P&L | Notable |
|-------|-----|---------|
| 2024-08 | +$1,142 | UHS +11.3% |
| 2024-09 | +$1,589 | PLTR +18.2% |
| 2024-10 | +$2,245 | CVNA +42.0%, APP +29.8% |
| 2024-11 | +$7,581 | APP +98.8% |
| 2024-12 | -$1,760 | 全面回調 |
| 2025-01 | +$4,602 | HOOD +39.4% |
| 2025-02 | -$1,153 | UAL -11.4%, APP -11.9% |
| 2025-03 | -$3,755 | HOOD -16.9%, UAL -26.4% |
| 2025-04 | +$3,659 | PLTR +40.3% |
| 2025-05 | +$4,227 | HOOD +34.7%, HWM +22.7% |
| 2025-06 | +$4,273 | HOOD +41.5% |
| 2025-07 | +$1,712 | GEV +24.8% |
| 2025-08 | -$2,432 | SMCI -29.6% |
| 2025-09 | +$6,014 | WDC +49.6%, HOOD +37.6% |
| 2025-10 | +$5,753 | SNDK +77.7% |
| 2025-11 | +$352 | Mixed |
| 2025-12 | -$843 | HOOD -12.0% |
| 2026-01 | +$10,441 | SNDK +142.8% |
| 2026-02 | +$1,263 | MRNA +21.6% |
| 2026-03 | -$1,836 | SNDK -17.0%, FIX -10.5% |

---

## 9. Conclusions & Improvement Directions

### Strengths

1. **正期望值 ($83.64/trade)**：策略在統計上有利可圖
2. **56.7% 勝率**：超過半數交易獲利
3. **1.79 Profit Factor**：每虧損 $1 可回收 $1.79
4. **IC 顯著**：動量因子對報酬有預測力

### Weaknesses

1. **ICIR 低 (0.15)**：預測力不穩定，某些月份 IC 為負（動量信號反向）
2. **Total Efficiency 低 (13.5%)**：月內價格區間捕獲率差
3. **66% 虧損交易曾獲利**：缺乏止盈機制導致大量利潤回吐
4. **MFE Captured 為負**：平均而言，實際報酬遠低於持有期間最佳可能

### Recommended Improvements

| Priority | Direction | Expected Impact |
|----------|-----------|-----------------|
| HIGH | 加入 trailing stop-profit（例如回撤 MFE 的 50% 即出場） | 減少利潤回吐，提升 Total Efficiency |
| HIGH | IC regime filter（IC 為負時降低倉位或暫停交易） | 提升 ICIR |
| MEDIUM | Sector filter（移除 Utilities/Materials 或降低權重） | 集中資金在高貢獻板塊 |
| MEDIUM | 縮短持有期（雙週或動態出場） | 更快鎖定利潤 |
| LOW | 動量因子增強（加入品質/波動率因子） | 改善選股精確度 |

---

## 10. Reproduction

```bash
# 安裝依賴
pip install -r requirements.txt

# 建立 Sector AUM 資料
python scripts/build_sector_aum_csv.py --years 20

# 執行交易分析（本報告使用的指令）
python scripts/trade_analysis.py --years 20
```

### Script Location

`scripts/trade_analysis.py` — 350 行，獨立於原有回測系統，專注於 per-trade 分析。

---

## Appendix: Metric Definitions

| Metric | Formula | Description |
|--------|---------|-------------|
| Expectancy | `(WR × AvgWin) - (LR × AvgLoss)` | 每筆交易平均期望獲利 |
| IC | `Spearman(momentum_scores, actual_returns)` | 預測排序與實際排序的相關性 |
| ICIR | `Mean(IC) / Std(IC)` | IC 穩定性指標 |
| Entry Efficiency | `(Highest - Entry) / (Highest - Lowest)` | 進場品質 |
| Exit Efficiency | `(Exit - Lowest) / (Highest - Lowest)` | 出場品質 |
| Total Efficiency | `(Exit - Entry) / (Highest - Lowest)` | 總捕獲效率 |
| MFE | `(Highest - Entry) / Entry` | 持有期間最大浮盈 |
| MAE | `(Lowest - Entry) / Entry` | 持有期間最大浮虧 |
| Profit Factor | `Sum(Wins) / Sum(Losses)` | 總獲利與總虧損比 |
