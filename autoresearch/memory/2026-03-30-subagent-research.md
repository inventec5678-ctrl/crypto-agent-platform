# Subagent Research - 2026-03-30

## Task
Find 2-3 new candidate strategies for the crypto quantitative trading system.

## Research Process

### 1. Data Analysis
- Analyzed 19 coins × 3 timeframes (1d, 4h, 15m)
- Data format: parquet files with OHLCV columns

### 2. Backtesting Approaches Tested
- Simple vectorized backtest (fast but limited)
- ATR trailing stop strategies
- RSI + EMA combinations
- Bollinger Band strategies
- Stochastic + RSI combos
- Death Cross variants

### 3. Key Findings

#### 4H Research (from 2026-03-27-4h-strategy.md)
- Best strategy: EMA(10,15) + RSI(16) + ATR Trailing Stop
- WR: 51.59%, PF: 2.25, DD: 24.99%, Sharpe: 2.30
- 157 trades over ~3 years
- Exit on reverse EMA cross

#### 15M Research (from 2026-03-27-15m-research.md)
- 15M timeframe is essentially non-tradable with pure technical analysis
- ATR ~0.33% per bar causes too many stop-outs
- Multiple strategy types tested (EMA, BB, Donchian, MACD, RSI, Ichimoku, etc.) - all failed

#### Research Loop Observations
- Recent experiments show very poor results (WR ~22%, PF ~1.4)
- Candlestick patterns (Three White Soldiers, Engulfing, PinBar) don't work on crypto
- The continuous research loop seems to be exploring too many random strategies

### 4. New Candidate Strategies Added

Based on research findings, 4 candidate strategies were added to factor_library.json:

1. **EMA_10_15_RSI_16_4H**
   - Based on: 4H research findings
   - Entry: EMA(10) crosses EMA(15), RSI(16) > 50 for long, < 50 for short
   - Exit: Reverse EMA cross with ATR trailing stop (1.0 ATR)
   - Research metrics: WR 51.59%, PF 2.25, DD 24.99%, Sharpe 2.30

2. **SHORT_DeathCross_4H**
   - Based on: SHORT_ONLY_DeathCross (1D) adapted to 4H
   - Entry: SMA20/50 death cross + close<MA200 + MA200_slope_negative + RSI<48 + momentum<0
   - SL: 3.5%, TP: 4.0%
   - Status: Needs backtesting

3. **EMA_8_40_RSI_55_1D**
   - Based on: EMA_Crossover_1D variant
   - Entry: EMA8 crosses EMA40, RSI>55 for long (tighter than original 50)
   - Exit: ATR-based (1.5 ATR SL, 2.5 ATR TP)
   - Status: Needs backtesting

4. **LONG_Breakout_4H**
   - Entry: EMA20/50 golden cross + RSI 45-65 + volume>1.2x + price>MA200
   - Exit: ATR-based (2.0 ATR SL, 3.0 ATR TP)
   - Status: Needs backtesting

### 5. Limitations

My backtest implementation differs from the production BacktestEngine in several ways:
- Proper ATR trailing stop implementation
- Position management (single position tracking)
- Trade execution on same-bar entry/exit

This likely explains why my backtest results don't match the verified strategy metrics.

### 6. Recommendations for Next Steps

1. **Verify new candidates** - Run proper backtests using AutoResearchClient
2. **Focus on 4H timeframe** - The 4H research shows promise (EMA 10/15 + RSI 16)
3. **Avoid 15M** - Research shows 15M is non-tradable with pure technical analysis
4. **Consider SHORT-ONLY variants** - Death Cross on 4H could be promising

### 7. Updated factor_library.json

- Total strategies: 9
- Verified: 5
- Candidate: 4
