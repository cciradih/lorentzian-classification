//+------------------------------------------------------------------+
//|                                                    Filters.mqh   |
//| Signal filters: volatility, regime, ADX, EMA/SMA trend.           |
//+------------------------------------------------------------------+
//
// ============================
// ==== Prediction Filters ====
// ============================
//
// User-Defined Filters: Used for adjusting the frequency of the ML model's
// predictions. There are two distinct layers of filters in this indicator:
//
//   (1) Signal-flip gates (volatility, regime, ADX). These form
//       filterAll = vol AND regime AND adx, which is required for the
//       ML prediction to flip the persistent signal state. Without them
//       agreeing, the signal latches to its previous value.
//
//   (2) Entry-level trend filters (EMA, SMA). These do not influence the
//       signal state but gate whether a signal flip is allowed to produce
//       an actual entry. Used to require price-above-EMA for longs and
//       price-below-EMA for shorts as additional trend confluence.
//
// Filter descriptions:
//   - Volatility Filter: passes when recentATR(1) > historicalATR(10).
//     Rejects low-volatility environments where the ML model is more
//     likely to generate noise.
//   - Regime Filter (Kaufman Adaptive Slope): uses a Kalman-like moving
//     filter (KLMF) over ohlc4 to estimate a trend curve, then compares
//     the curve's absolute slope to an EMA(200) of that slope. A
//     user-defined threshold on the normalized slope distinguishes
//     trending vs ranging markets. Negative thresholds (default -0.1)
//     tolerate slightly-below-average trendiness.
//   - ADX Filter: passes when ADX > threshold, requiring a minimum
//     directional-movement strength for signal flips. This is an
//     independent ADX computation -- it does NOT reuse the ADX feature
//     (which is rescaled to [0,1]) and instead computes the raw 0-100
//     ADX internally.
//   - EMA / SMA Filters: classical moving-average trend filters. When
//     enabled, longs require close > MA and shorts require close < MA.
//     Default period is 200. When disabled, the uptrend/downtrend getters
//     return true so they never block entries.
//
#ifndef __LC_FILTERS_MQH__
#define __LC_FILTERS_MQH__

#include "MLFeatures.mqh"

//+------------------------------------------------------------------+
//| Filter working state (all arrays caller-allocated and resized)   |
//+------------------------------------------------------------------+
struct FilterState
{
   // Volatility: ATR(1) and ATR(10)
   double volTR1[];
   double volATR1[];       // ATR period 1
   double volTR10[];
   double volATR10[];      // ATR period 10

   // Regime filter
   double regValue1[];
   double regValue2[];
   double regKLMF[];
   double regAbsSlope[];
   double regEmaAbsSlope[];

   // ADX filter (independent from feature ADX)
   double adxTR[], adxDMPlus[], adxDMMinus[];
   double adxTRSmooth[], adxSmoothDMPlus[], adxSmoothDMMinus[];
   double adxDX[], adxRMA[], adxValue[];

   // EMA / SMA trend filters
   double emaArr[];
   double smaArr[];
};

void ResizeFilterState(FilterState &fs, int total)
{
   ArrayResize(fs.volTR1, total);     ArrayResize(fs.volATR1, total);
   ArrayResize(fs.volTR10, total);    ArrayResize(fs.volATR10, total);
   ArrayResize(fs.regValue1, total);  ArrayResize(fs.regValue2, total);
   ArrayResize(fs.regKLMF, total);    ArrayResize(fs.regAbsSlope, total);
   ArrayResize(fs.regEmaAbsSlope, total);
   ArrayResize(fs.adxTR, total);      ArrayResize(fs.adxDMPlus, total);
   ArrayResize(fs.adxDMMinus, total); ArrayResize(fs.adxTRSmooth, total);
   ArrayResize(fs.adxSmoothDMPlus, total); ArrayResize(fs.adxSmoothDMMinus, total);
   ArrayResize(fs.adxDX, total);      ArrayResize(fs.adxRMA, total);
   ArrayResize(fs.adxValue, total);
   ArrayResize(fs.emaArr, total);     ArrayResize(fs.smaArr, total);
}

//+------------------------------------------------------------------+
//| Volatility Filter: recentATR(1) > historicalATR(10)              |
//|                                                                  |
//| Passes when the most recent bar's true range exceeds the 10-bar  |
//| ATR. The intent is to accept bars with meaningful volatility and |
//| reject stalled/contracted periods where the ML model's signal is |
//| more likely to be noise.                                         |
//|                                                                  |
//| During warmup (ATR values not yet valid) the filter returns true |
//| so the indicator isn't blocked from producing early signals.     |
//+------------------------------------------------------------------+
void CalcVolatilityFilter(const double &high[], const double &low[],
                          const double &close[], FilterState &fs,
                          int begin, int total)
{
   CalcATR(high, low, close, 1,  fs.volATR1,  fs.volTR1,  begin, total);
   CalcATR(high, low, close, 10, fs.volATR10, fs.volTR10, begin, total);
}

bool GetVolatilityFilter(const FilterState &fs, int i, bool useFilter)
{
   if(!useFilter) return true;
   if(fs.volATR1[i] == LC_EMPTY || fs.volATR10[i] == LC_EMPTY) return true;
   return fs.volATR1[i] > fs.volATR10[i];
}

//+------------------------------------------------------------------+
//| Regime Filter (Kaufman adaptive slope)                           |
//|                                                                  |
//| Uses a Kalman-Like Moving Filter (KLMF) over ohlc4 to build an   |
//| adaptive trend estimate, then measures the curve's absolute      |
//| slope. A user-defined threshold on the slope's deviation from    |
//| its EMA(200) distinguishes trending from ranging conditions.     |
//|                                                                  |
//| State variables:                                                 |
//|   value1  : EMA-like tracker of ohlc4 first difference (drift).  |
//|   value2  : EMA-like tracker of high-low range (noise floor).    |
//|   omega   : |value1/value2| (signal-to-noise proxy).             |
//|   alpha   : adaptive gain derived from omega.                    |
//|   KLMF    : alpha * ohlc4 + (1-alpha) * prev; fast when trending,|
//|             slow when choppy.                                    |
//|   slope   : |KLMF[i] - KLMF[i-1]|                                |
//|                                                                  |
//| Pass condition: (slope - EMA(slope, 200)) / EMA(slope, 200) >=   |
//| threshold. A negative threshold (default -0.1) accepts slightly- |
//| below-average trendiness; higher values require stronger trend.  |
//+------------------------------------------------------------------+
void CalcRegimeFilter(const double &ohlc4[], const double &high[],
                      const double &low[], FilterState &fs,
                      int begin, int total)
{
   // Recursive state: must process bar-by-bar
   if(begin == 0)
   {
      fs.regValue1[0] = 0;
      fs.regValue2[0] = high[0] - low[0];
      fs.regKLMF[0]   = ohlc4[0];
      fs.regAbsSlope[0]    = 0;
      fs.regEmaAbsSlope[0] = 0;
      begin = 1;
   }
   for(int i = begin; i < total; i++)
   {
      double prevV1   = NZ(fs.regValue1[i - 1]);
      double prevV2   = NZ(fs.regValue2[i - 1]);
      double prevKLMF = NZ(fs.regKLMF[i - 1]);

      fs.regValue1[i] = 0.2 * (ohlc4[i] - ohlc4[i - 1]) + 0.8 * prevV1;
      fs.regValue2[i] = 0.1 * (high[i] - low[i]) + 0.8 * prevV2;

      double omega = fs.regValue2[i] != 0 ? MathAbs(fs.regValue1[i] / fs.regValue2[i]) : 0;
      double alpha = (-MathPow(omega, 2) +
                      MathSqrt(MathPow(omega, 4) + 16.0 * MathPow(omega, 2))) / 8.0;
      fs.regKLMF[i] = alpha * ohlc4[i] + (1.0 - alpha) * prevKLMF;

      fs.regAbsSlope[i] = MathAbs(fs.regKLMF[i] - fs.regKLMF[i - 1]);

      // EMA of absolute curve slope (period 200)
      double emaAlpha = 2.0 / (200 + 1);
      double prevEma = NZ(fs.regEmaAbsSlope[i - 1]);
      if(prevEma == 0 && i < 200)
         fs.regEmaAbsSlope[i] = fs.regAbsSlope[i]; // bootstrap
      else
         fs.regEmaAbsSlope[i] = emaAlpha * fs.regAbsSlope[i] + (1.0 - emaAlpha) * prevEma;
   }
}

bool GetRegimeFilter(const FilterState &fs, int i, double threshold, bool useFilter)
{
   if(!useFilter) return true;
   if(fs.regEmaAbsSlope[i] == 0) return true;
   double normSlope = (fs.regAbsSlope[i] - fs.regEmaAbsSlope[i]) / fs.regEmaAbsSlope[i];
   return normSlope >= threshold;
}

//+------------------------------------------------------------------+
//| ADX Filter: adx > threshold                                     |
//| Independent ADX computation (not reusing feature ADX).           |
//|                                                                  |
//| Passes when the 14-period Wilder ADX exceeds the user threshold. |
//| Defaults: period=14, threshold=20. Used as an additional         |
//| directional-movement gate on top of the volatility/regime gates. |
//|                                                                  |
//| This is computed on raw 0-100 ADX, whereas FEATURE_ADX (used as  |
//| an ML input) is additionally rescaled to [0,1]. The two cannot   |
//| share state because they have different output ranges and may    |
//| use different periods.                                           |
//+------------------------------------------------------------------+
void CalcADXFilter(const double &high[], const double &low[],
                   const double &close[], int period, FilterState &fs,
                   int begin, int total)
{
   // Bar 0 has no prior bar; prev close/high/low default to 0.
   for(int i = begin; i < total; i++)
   {
      double prevClose = (i > 0) ? close[i - 1] : 0;
      double prevHigh  = (i > 0) ? high[i - 1]  : 0;
      double prevLow   = (i > 0) ? low[i - 1]   : 0;
      fs.adxTR[i] = MathMax(MathMax(high[i] - low[i],
                     MathAbs(high[i] - prevClose)),
                     MathAbs(low[i] - prevClose));
      double up   = high[i] - prevHigh;
      double down = prevLow - low[i];
      fs.adxDMPlus[i]  = (up > down && up > 0)     ? up   : 0;
      fs.adxDMMinus[i] = (down > up && down > 0)    ? down : 0;
   }
   CalcWilderSmooth(fs.adxTR, period, fs.adxTRSmooth, begin, total);
   CalcWilderSmooth(fs.adxDMPlus, period, fs.adxSmoothDMPlus, begin, total);
   CalcWilderSmooth(fs.adxDMMinus, period, fs.adxSmoothDMMinus, begin, total);
   for(int i = begin; i < total; i++)
   {
      double diP = fs.adxTRSmooth[i] != 0 ? fs.adxSmoothDMPlus[i] / fs.adxTRSmooth[i] * 100 : 0;
      double diN = fs.adxTRSmooth[i] != 0 ? fs.adxSmoothDMMinus[i] / fs.adxTRSmooth[i] * 100 : 0;
      fs.adxDX[i] = (diP + diN) != 0 ? MathAbs(diP - diN) / (diP + diN) * 100 : 0;
   }
   CalcRMA(fs.adxDX, period, fs.adxRMA, begin, total);
   for(int i = begin; i < total; i++)
      fs.adxValue[i] = NZ(fs.adxRMA[i]);
}

bool GetADXFilter(const FilterState &fs, int i, int threshold, bool useFilter)
{
   if(!useFilter) return true;
   return fs.adxValue[i] > threshold;
}

//+------------------------------------------------------------------+
//| EMA / SMA Trend Filters                                          |
//|                                                                  |
//| Classical moving-average trend filters. When enabled:            |
//|   uptrend   := close > MA                                        |
//|   downtrend := close < MA                                        |
//| Longs require uptrend; shorts require downtrend. When disabled,  |
//| both getters return true so the filter does not block entries.   |
//| Default period: 200 for both. These filters apply at the entry   |
//| stage and do NOT participate in the signal-flip gate (filterAll).|
//+------------------------------------------------------------------+
void CalcEMAFilter(const double &close[], int period, FilterState &fs,
                   int begin, int total)
{
   CalcEMA(close, period, fs.emaArr, begin, total);
}

void CalcSMAFilter(const double &close[], int period, FilterState &fs,
                   int begin, int total)
{
   CalcSMA(close, period, fs.smaArr, begin, total);
}

bool GetEMAUptrend(const FilterState &fs, const double &close[], int i, bool useFilter)
{
   if(!useFilter) return true;
   return fs.emaArr[i] != LC_EMPTY && close[i] > fs.emaArr[i];
}

bool GetEMADowntrend(const FilterState &fs, const double &close[], int i, bool useFilter)
{
   if(!useFilter) return true;
   return fs.emaArr[i] != LC_EMPTY && close[i] < fs.emaArr[i];
}

bool GetSMAUptrend(const FilterState &fs, const double &close[], int i, bool useFilter)
{
   if(!useFilter) return true;
   return fs.smaArr[i] != LC_EMPTY && close[i] > fs.smaArr[i];
}

bool GetSMADowntrend(const FilterState &fs, const double &close[], int i, bool useFilter)
{
   if(!useFilter) return true;
   return fs.smaArr[i] != LC_EMPTY && close[i] < fs.smaArr[i];
}

#endif // __LC_FILTERS_MQH__
