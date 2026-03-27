# Strategy Playbook - S&P 500 Sector Momentum (Research Version)

本文件是本專案的「研究級」策略框架說明，從研究假設、資料建模、策略設計、回測流程、偏誤控制、穩健性驗證到上線前審核，完整描述如何從零實作到可審計的結果。

適用範圍：

- 策略研究與回測 (`scripts/full_backtest.py`, `scripts/walk_forward.py`, `scripts/sensitivity_backtest.py`)
- 資料準備與權重建構 (`data_manager.py`, `scripts/build_sector_aum_csv.py`)
- API 回測 (`services/backtest_service.py`)

---

## 1. 研究問題與假設

### 1.1 研究問題

在 S&P 500 可投資 universe 下，能否透過「板塊配置 + 板塊內動量選股」在長期（月頻）取得相對 SPY / SPMO / SPX 的風險調整後超額報酬。

### 1.2 核心假設

1. **跨板塊相對強弱**具持續性：同時期不同板塊的資金偏好具有中期延續。
2. **板塊內動量**具可交易性：在單一板塊中，強勢股於後續月度窗口仍有優勢。
3. **月頻再平衡**可在交易成本可控下，捕捉趨勢而不過度交易。

### 1.3 可反駁條件

若下列任一條件長期成立，策略即不具研究價值：

- OOS（walk-forward）大多數窗口為負或 alpha 不顯著。
- 成本上調後（例如 2x）績效大幅崩潰。
- 參數微調後結果高度不穩定（尖峰型參數敏感度）。

---

## 2. 策略總覽（Framework）

### 2.1 模組架構

- `data_manager.py`：下載與整理 universe、價格、benchmark、sector 權重。
- `strategy.py`：計算動量、按板塊選股、生成目標權重。
- `backtester.py`：月度再平衡、成本扣除、績效路徑與交易紀錄。
- `analyzer.py`：績效指標與報告整合。
- `scripts/*.py`：研究入口（full / walk-forward / sensitivity / AUM build）。

### 2.2 計算流程（高階）

1. 準備資料：PIT universe、股價、板塊權重、benchmark。
2. 動量計算：每股計算 4W/13W/26W 回報並平均。
3. 月度選股：每板塊選 Top-N。
4. 權重組合：先板塊，再板塊內均分到股票。
5. 下期執行：`signal_lag_months=1`，於下一個月生效。
6. 扣除成本：turnover 乘交易成本與滑價。
7. 輸出報告：總績效、風險、OOS、敏感度。

---

## 3. 資料工程設計

### 3.1 Universe（避免存活者偏誤）

策略不直接使用「今日 S&P 500 名單回測全歷史」。

- 研究預設：`use_point_in_time_universe=True`
- 做法：按月重建當期可投資名單，並在回測當月使用對應 universe。

### 3.2 價格資料

- 股票價格：主要透過 yfinance 下載並對齊。
- benchmark：SPY / SPMO / SPX。
- 月度化：日資料轉 `ME`（month-end）以匹配再平衡節奏。

### 3.3 板塊權重資料（關鍵）

本專案已採「回測優先」規範：

- 回測預設要求本地檔 `data/sector_aum_monthly.csv`
- 設定：`config.py` 中 `REQUIRE_LOCAL_SECTOR_AUM_FOR_BACKTEST = True`

權重來源優先順序：

1. 真實本地 CSV（首選）
2. `sharesOutstanding * close` 生成月度近似
3. proxy（current AUM + historical price scaling）僅作 fallback

> 研究建議：正式研究報告避免使用第 3 種，除非明確標註其前視風險。

---

## 4. 訊號設計與數學定義

### 4.1 動量分數

對股票 `i`、日期 `t`：

```text
Ret_21(i,t)  = P(i,t) / P(i,t-21)  - 1
Ret_63(i,t)  = P(i,t) / P(i,t-63)  - 1
Ret_126(i,t) = P(i,t) / P(i,t-126) - 1

Momentum(i,t) = (Ret_21 + Ret_63 + Ret_126) / 3
```

### 4.2 板塊內選股

對每個 sector `s`：

1. 取該 sector 當期 universe 的可交易股票集合。
2. 依 `Momentum(i,t)` 由大到小排序。
3. 取前 `N = TOP_N_PER_SECTOR` 檔。

### 4.3 權重分配

令 `W_sector(s,t)` 為當月板塊權重，`K_s` 為該板塊入選股票數。

```text
W_stock(i,t) = W_sector(s,t) / K_s , i ∈ sector s selected set
```

若板塊該月無有效標的，該權重不強制重分配（可形成現金占比）。

---

## 5. 交易模擬與執行假設

### 5.1 再平衡頻率

- 月度再平衡（month-end 節點）

### 5.2 訊號執行延遲

- `execution_lag_months = 1`
- 代表：用 `t` 月訊號，於 `t+1` 月執行，以降低同 bar 前視問題。

### 5.3 交易成本模型

單月成本以 turnover 為基礎：

```text
turnover_t = 0.5 * Σ_i |w_target(i,t) - w_prev(i,t)|
cost_t = turnover_t * (transaction_cost + slippage_cost)
net_return_t = gross_return_t - cost_t
```

預設參數：

- `transaction_cost = 0.001`
- `slippage_cost = 0.0005`

### 5.4 暖機期（Warm-up）

- 前 7 個月不交易（確保 126d 動量視窗完整）。

---

## 6. 回測實驗設計

### 6.1 Full Backtest（歷史全段）

用途：觀察長期表現、風險輪廓、基準比較。

```bash
python build_sector_aum_csv.py --years 20
python full_backtest.py --years 20 --no-plots
```

輸出：

- `reports/backtest_metrics.csv`
- `reports/backtest_report.png`（若未 `--no-plots`）

### 6.2 Walk-forward（OOS 穩定性）

用途：檢查不同市場 regime 下，樣本外是否穩定。

```bash
python walk_forward.py --years 20 --train-months 36 --test-months 12
```

輸出：`reports/walk_forward_report.csv`

### 6.3 Sensitivity（參數穩健性）

用途：檢查策略是否只在單一參數點有效。

```bash
python sensitivity_backtest.py --years 20
```

輸出：`reports/sensitivity_report.csv`

---

## 7. 風險指標與評估框架

### 7.1 主要指標

- Total Return
- CAGR
- Annualized Volatility
- Sharpe / Sortino / Calmar
- Max Drawdown (MDD)
- MDD Duration
- Monthly Win Rate

### 7.2 審核邏輯（建議）

1. **收益面**：CAGR 是否顯著高於基準。
2. **風險面**：MDD 與回復時間是否在可承受範圍。
3. **穩健面**：walk-forward 負窗口比例是否可接受。
4. **脆弱度**：成本上調、參數偏移後是否仍有 alpha。

---

## 8. 偏誤與風控（研究審計重點）

### 8.1 存活者偏誤（Survivorship Bias）

- 風險：只用當前成分股回測歷史。
- 控制：PIT universe + 月度名單回推。

### 8.2 前視偏誤（Look-ahead Bias）

- 風險：使用未來才知道的資料（例如 current AUM 回推歷史）。
- 控制：
  - 回測預設要求本地 `sector_aum_monthly.csv`
  - `execution_lag_months=1`
  - 報告需標示 AUM 來源（shares_x_price / proxy）

### 8.3 成交可行性偏誤

- 風險：忽略市場衝擊、流動性分層。
- 控制：提高成本情境測試（2x/3x），觀察 alpha 存活性。

---

## 9. 研究流程 SOP（從頭到尾）

### Step 0 - 環境固定

```bash
pip install -r requirements.txt
python test_all.py
```

### Step 1 - 建立回測資料快照

```bash
python build_sector_aum_csv.py --years 20
```

若要嚴格禁止 proxy：

```bash
python build_sector_aum_csv.py --years 20 --no-proxy-fallback
```

### Step 2 - 跑基準回測

```bash
python full_backtest.py --years 20 --no-plots
```

### Step 3 - 做 OOS 穩健性

```bash
python walk_forward.py --years 20 --train-months 36 --test-months 12
```

### Step 4 - 做敏感度壓力測試

```bash
python sensitivity_backtest.py --years 20
```

### Step 5 - 審核與結論

檢閱：

- `reports/backtest_metrics.csv`
- `reports/walk_forward_report.csv`
- `reports/sensitivity_report.csv`

並回答：

1. OOS 是否仍有優勢？
2. 成本提高後是否可活？
3. 回撤是否符合實盤承受能力？

---

## 10. 報告撰寫模板（建議章節）

1. 研究動機與問題定義
2. 資料來源與清洗流程
3. 策略定義與公式
4. 回測設計（樣本、頻率、成本、延遲）
5. 主要結果（含基準比較）
6. 穩健性（walk-forward + sensitivity）
7. 偏誤分析與控制措施
8. 風險揭露與失效條件
9. 上線建議與下一步

---

## 11. 上線前 Go / No-Go 清單

- `GO` 條件（至少滿足）：
  - PIT universe 啟用且可重現
  - 20 年回測 + OOS + sensitivity 三份報告一致支持
  - 成本壓力測試後仍保有正向 alpha
  - 風險（MDD / 回復期）在授權範圍

- `NO-GO` 條件（任一觸發）：
  - 依賴 proxy AUM 且未揭露
  - OOS 主要窗口為負
  - 參數微調即失效
  - 交易成本稍上調即失去優勢

---

## 12. 限制與後續研究方向

### 12.1 目前限制

- yfinance shares 歷史在部分 ETF 可能缺值。
- PIT 資料屬高可用近似，不是商業級官方 PIT feed。
- 月頻成本模型仍是簡化，未含完整 market impact。

### 12.2 後續強化

1. 引入更完整歷史 AUM / shares 資料源。
2. 加入流動性分層（ADV、成交額門檻）與衝擊成本模型。
3. 增加 regime-aware 風險控制（波動目標、風險平價約束）。
4. 建立固定版本的研究資料快照，支持完全可重演。

---

## 13. 快速命令索引

```bash
# 1) 建立本地 AUM
python build_sector_aum_csv.py --years 20

# 2) 全歷史回測
python full_backtest.py --years 20 --no-plots

# 3) OOS 驗證
python walk_forward.py --years 20 --train-months 36 --test-months 12

# 4) 參數壓力測試
python sensitivity_backtest.py --years 20

# 5) 全量測試
python test_all.py
```
