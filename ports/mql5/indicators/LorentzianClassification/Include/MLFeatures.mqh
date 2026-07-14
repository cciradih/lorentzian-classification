//+------------------------------------------------------------------+
//|                                                  MLFeatures.mqh  |
//| Feature-engineering primitives and technical indicators.          |
//+------------------------------------------------------------------+
//
// =============================
// ==== Feature Engineering ====
// =============================
//
// The ML model consumes up to 5 features per bar. In reality, the optimal
// number of features is in the range of 3-8. Each feature is an oscillator
// mapped into a bounded range so that comparisons across different
// indicators are numerically meaningful. This is the essence of feature
// normalization: the ML model's distance calculation treats every feature
// as equally weighted, so without a common range the feature with the
// largest raw magnitude would dominate the distance.
//
// Feature primitives supported:
//   - FEATURE_RSI : Normalized RSI. rescale(ema(rsi(src, n1), n2), 0,100, 0,1).
//                   ParamA = RSI period, ParamB = EMA smoothing period.
//                   A classical momentum oscillator bounded to [0,1].
//   - FEATURE_WT  : Normalized WaveTrend. Two-stage EMA over TCI / absDev,
//                   then min-max normalized to [0,1] via running extremes.
//                   ParamA = n1 (channel length), ParamB = n2 (average).
//   - FEATURE_CCI : Normalized CCI. ema-smoothed CCI min-max normalized
//                   to [0,1]. ParamA = CCI period, ParamB = EMA smoothing.
//   - FEATURE_ADX : Average Directional Index, rescaled from [0,100] to
//                   [0,1]. ParamA = ADX period. ParamB is unused.
//
// Numerical conventions:
//   - All working arrays are forward-indexed (0 = oldest bar).
//   - LC_EMPTY (= EMPTY_VALUE) is the sentinel for warmup / missing values.
//     Any arithmetic involving an LC_EMPTY operand propagates LC_EMPTY
//     downstream so that dependent calculations stay invalid until enough
//     history is available.
//   - NZ(val, fallback=0) substitutes LC_EMPTY / NaN with fallback.
//   - normalize() is min-max scaling using running historic extremes
//     (state carried across bars). rescale() is a fixed-range linear map
//     used when the input's bounds are known a priori (e.g. RSI in 0..100).
//
#ifndef __LC_ML_FEATURES_MQH__
#define __LC_ML_FEATURES_MQH__

// Sentinel for missing values
#define LC_EMPTY EMPTY_VALUE

//+------------------------------------------------------------------+
//| NZ - substitute LC_EMPTY / NaN with a fallback (default 0)       |
//+------------------------------------------------------------------+
double NZ(double val, double fallback=0.0)
{
   return (val == LC_EMPTY || !MathIsValidNumber(val)) ? fallback : val;
}

//+------------------------------------------------------------------+
//| Normalization state (tracks running min/max per call site)        |
//+------------------------------------------------------------------+
struct NormState
{
   double historicMin;
   double historicMax;
};

void InitNormState(NormState &state)
{
   state.historicMin = 10e10;
   state.historicMax = -10e10;
}

// Min-max scale into [outMin, outMax] using running historic extremes
double ApplyNormalize(double src, double outMin, double outMax, NormState &state)
{
   if(src != LC_EMPTY && MathIsValidNumber(src))
   {
      if(src < state.historicMin) state.historicMin = src;
      if(src > state.historicMax) state.historicMax = src;
   }
   double range = MathMax(state.historicMax - state.historicMin, 10e-10);
   return outMin + (outMax - outMin) * (src - state.historicMin) / range;
}

// Linear map from one bounded range to another
double ApplyRescale(double src, double oldMin, double oldMax, double newMin, double newMax)
{
   double range = MathMax(oldMax - oldMin, 10e-10);
   return newMin + (newMax - newMin) * (src - oldMin) / range;
}

//+------------------------------------------------------------------+
//| SMA - Simple Moving Average                                      |
//| All arrays are forward-indexed: index 0 = oldest bar             |
//+------------------------------------------------------------------+
void CalcSMA(const double &src[], int period, double &out[], int begin, int total)
{
   if(period <= 0) { for(int i = begin; i < total; i++) out[i] = LC_EMPTY; return; }
   for(int i = begin; i < total; i++)
   {
      if(i < period - 1) { out[i] = LC_EMPTY; continue; }
      double sum = 0;
      bool allValid = true;
      for(int j = i - period + 1; j <= i; j++)
      {
         if(src[j] == LC_EMPTY) { allValid = false; break; }
         sum += src[j];
      }
      out[i] = allValid ? sum / period : LC_EMPTY;
   }
}

//+------------------------------------------------------------------+
//| EMA - Exponential Moving Average                                  |
//| Seeds with SMA on first valid bar, then alpha recurrence.        |
//| alpha = 2 / (period + 1)                                        |
//+------------------------------------------------------------------+
void CalcEMA(const double &src[], int period, double &out[], int begin, int total)
{
   if(period <= 0) { for(int i = begin; i < total; i++) out[i] = LC_EMPTY; return; }
   double alpha = 2.0 / (period + 1);
   for(int i = begin; i < total; i++)
   {
      // Invalid source -> invalid output
      if(src[i] == LC_EMPTY) { out[i] = LC_EMPTY; continue; }
      // Valid previous EMA -> normal recurrence
      if(i > 0 && out[i - 1] != LC_EMPTY)
      {
         out[i] = alpha * src[i] + (1.0 - alpha) * out[i - 1];
         continue;
      }
      // Try to seed with SMA of last `period` bars
      if(i >= period - 1)
      {
         double sum = 0;
         bool allValid = true;
         for(int j = i - period + 1; j <= i; j++)
         {
            if(src[j] == LC_EMPTY) { allValid = false; break; }
            sum += src[j];
         }
         out[i] = allValid ? sum / period : LC_EMPTY;
      }
      else
         out[i] = LC_EMPTY;
   }
}

//+------------------------------------------------------------------+
//| RMA - Wilder's Smoothing (alpha = 1 / period, SMA seed)          |
//+------------------------------------------------------------------+
void CalcRMA(const double &src[], int period, double &out[], int begin, int total)
{
   if(period <= 0) { for(int i = begin; i < total; i++) out[i] = LC_EMPTY; return; }
   double alpha = 1.0 / period;
   for(int i = begin; i < total; i++)
   {
      if(src[i] == LC_EMPTY) { out[i] = LC_EMPTY; continue; }
      if(i > 0 && out[i - 1] != LC_EMPTY)
      {
         out[i] = alpha * src[i] + (1.0 - alpha) * out[i - 1];
         continue;
      }
      if(i >= period - 1)
      {
         double sum = 0;
         bool allValid = true;
         for(int j = i - period + 1; j <= i; j++)
         {
            if(src[j] == LC_EMPTY) { allValid = false; break; }
            sum += src[j];
         }
         out[i] = allValid ? sum / period : LC_EMPTY;
      }
      else
         out[i] = LC_EMPTY;
   }
}

//+------------------------------------------------------------------+
//| Wilder Directional Smoothing (no SMA seed - starts from 0)      |
//| Used in ADX: smooth = prev - prev/period + value                 |
//| On bar 0: nz(prev)=0, so result = value.                        |
//+------------------------------------------------------------------+
void CalcWilderSmooth(const double &src[], int period, double &out[], int begin, int total)
{
   if(period <= 0) { for(int i = begin; i < total; i++) out[i] = LC_EMPTY; return; }
   for(int i = begin; i < total; i++)
   {
      if(i == 0 || out[i - 1] == LC_EMPTY)
         out[i] = src[i];
      else
         out[i] = out[i - 1] - out[i - 1] / period + src[i];
   }
}

//+------------------------------------------------------------------+
//| RSI - Relative Strength Index                                    |
//| gain/loss/gainRMA/lossRMA are working arrays (caller-allocated)  |
//+------------------------------------------------------------------+
void CalcRSI(const double &src[], int period, double &out[],
             double &gain[], double &loss[],
             double &gainRMA[], double &lossRMA[],
             int begin, int total)
{
   // Gain/loss series (bar 0 has no prior bar -> LC_EMPTY)
   int start = MathMax(begin, 1);
   if(begin == 0) { gain[0] = LC_EMPTY; loss[0] = LC_EMPTY; }
   for(int i = start; i < total; i++)
   {
      double change = src[i] - src[i - 1];
      gain[i] = change > 0 ? change : 0;
      loss[i] = change < 0 ? -change : 0;
   }
   // RMA of gain and loss
   CalcRMA(gain, period, gainRMA, begin, total);
   CalcRMA(loss, period, lossRMA, begin, total);
   // RSI = 100 - 100/(1 + rs)
   for(int i = begin; i < total; i++)
   {
      if(gainRMA[i] == LC_EMPTY || lossRMA[i] == LC_EMPTY)
      { out[i] = LC_EMPTY; continue; }
      if(lossRMA[i] == 0) { out[i] = 100.0; continue; }
      double rs = gainRMA[i] / lossRMA[i];
      out[i] = 100.0 - 100.0 / (1.0 + rs);
   }
}

//+------------------------------------------------------------------+
//| CCI - Commodity Channel Index                                    |
//| smaArr is a working array (caller-allocated)                     |
//+------------------------------------------------------------------+
void CalcCCI(const double &src[], int period, double &out[],
             double &smaArr[], int begin, int total)
{
   CalcSMA(src, period, smaArr, begin, total);
   for(int i = begin; i < total; i++)
   {
      if(smaArr[i] == LC_EMPTY) { out[i] = LC_EMPTY; continue; }
      double sumDev = 0;
      for(int j = i - period + 1; j <= i; j++)
         sumDev += MathAbs(src[j] - smaArr[i]);
      double meanDev = sumDev / period;
      out[i] = meanDev != 0 ? (src[i] - smaArr[i]) / (0.015 * meanDev) : 0;
   }
}

//+------------------------------------------------------------------+
//| ATR - Average True Range (RMA of True Range)                     |
//| tr is a working array (caller-allocated)                         |
//+------------------------------------------------------------------+
void CalcATR(const double &high[], const double &low[], const double &close[],
             int period, double &out[], double &tr[],
             int begin, int total)
{
   // Bar 0 has no prior bar; previous close defaults to 0.
   for(int i = begin; i < total; i++)
   {
      double prevClose = (i > 0) ? close[i - 1] : 0;
      tr[i] = MathMax(MathMax(high[i] - low[i],
                               MathAbs(high[i] - prevClose)),
                       MathAbs(low[i] - prevClose));
   }
   CalcRMA(tr, period, out, begin, total);
}

//+------------------------------------------------------------------+
//| Quantized price subtraction (BigDecimal-equivalent).              |
//| The reference implementation stores prices via BigDecimal, so      |
//| `1.10580 - 1.10430` and `1.09670 - 1.09520` are both exactly       |
//| 0.00150 and compare equal. In float64, those subtractions differ   |
//| by 1 ULP, which can flip strict-> tie-breakers in +DM/-DM and      |
//| inject visible drift through Wilder smoothing. This helper rounds  |
//| to the integer-tick grid implied by `scale` (= 10^digits), does    |
//| the subtraction in long-space (exact), then divides back to float. |
//+------------------------------------------------------------------+
double QuantSub(double a, double b, double scale)
{
   if(scale <= 0.0) return a - b;
   long ai = (long)MathRound(a * scale);
   long bi = (long)MathRound(b * scale);
   return (double)(ai - bi) / scale;
}

//+------------------------------------------------------------------+
//| ADX - Average Directional Index                                  |
//| Uses Wilder directional smoothing (not SMA-seeded RMA).          |
//| Output is rescaled to [0, 1] for use as an ML feature.           |
//| All working arrays are caller-allocated.                         |
//| `priceScale` = 10^digits (e.g. 1e5 for EURUSD with 5 digits).    |
//| Pass 0.0 to disable tick-quantization (legacy behavior).         |
//+------------------------------------------------------------------+
void CalcADX(const double &high[], const double &low[], const double &close[],
             int period, double &out[],
             double &tr[], double &dmPlus[], double &dmMinus[],
             double &trSmooth[], double &smoothDMPlus[], double &smoothDMMinus[],
             double &dx[], double &adxRMA[],
             int begin, int total, double priceScale = 0.0)
{
   // Bar 0 has no prior bar; prev close/high/low default to 0.
   for(int i = begin; i < total; i++)
   {
      double prevClose = (i > 0) ? close[i - 1] : 0;
      double prevHigh  = (i > 0) ? high[i - 1]  : 0;
      double prevLow   = (i > 0) ? low[i - 1]   : 0;
      tr[i] = MathMax(MathMax(QuantSub(high[i], low[i], priceScale),
                               MathAbs(QuantSub(high[i], prevClose, priceScale))),
                       MathAbs(QuantSub(low[i], prevClose, priceScale)));
      double upMove   = QuantSub(high[i], prevHigh, priceScale);
      double downMove = QuantSub(prevLow, low[i], priceScale);
      dmPlus[i]  = (upMove > downMove && upMove > 0)   ? upMove   : 0;
      dmMinus[i] = (downMove > upMove && downMove > 0)  ? downMove : 0;
   }
   CalcWilderSmooth(tr, period, trSmooth, begin, total);
   CalcWilderSmooth(dmPlus, period, smoothDMPlus, begin, total);
   CalcWilderSmooth(dmMinus, period, smoothDMMinus, begin, total);
   for(int i = begin; i < total; i++)
   {
      double diP = trSmooth[i] != 0 ? smoothDMPlus[i] / trSmooth[i] * 100 : 0;
      double diN = trSmooth[i] != 0 ? smoothDMMinus[i] / trSmooth[i] * 100 : 0;
      dx[i] = (diP + diN) != 0 ? MathAbs(diP - diN) / (diP + diN) * 100 : 0;
   }
   CalcRMA(dx, period, adxRMA, begin, total);
   for(int i = begin; i < total; i++)
      out[i] = adxRMA[i] != LC_EMPTY ? ApplyRescale(adxRMA[i], 0, 100, 0, 1) : LC_EMPTY;
}

//+------------------------------------------------------------------+
//| WaveTrend (normalized)                                           |
//| ema1..raw are working arrays, normState is per-feature.          |
//| Output is normalized to [0, 1].                                  |
//+------------------------------------------------------------------+
void CalcWaveTrend(const double &src[], int n1, int n2, double &out[],
                   double &ema1[], double &absDev[], double &ema2[],
                   double &ci[], double &wt1[], double &wt2[], double &raw[],
                   NormState &normState, int begin, int total)
{
   CalcEMA(src, n1, ema1, begin, total);
   for(int i = begin; i < total; i++)
      absDev[i] = ema1[i] != LC_EMPTY ? MathAbs(src[i] - ema1[i]) : LC_EMPTY;
   CalcEMA(absDev, n1, ema2, begin, total);
   for(int i = begin; i < total; i++)
   {
      if(ema1[i] == LC_EMPTY || ema2[i] == LC_EMPTY) { ci[i] = LC_EMPTY; continue; }
      ci[i] = ema2[i] != 0 ? (src[i] - ema1[i]) / (0.015 * ema2[i]) : 0;
   }
   CalcEMA(ci, n2, wt1, begin, total);
   CalcSMA(wt1, 4, wt2, begin, total);
   for(int i = begin; i < total; i++)
   {
      if(wt1[i] == LC_EMPTY || wt2[i] == LC_EMPTY) { out[i] = LC_EMPTY; continue; }
      raw[i] = wt1[i] - wt2[i];
      out[i] = ApplyNormalize(raw[i], 0, 1, normState);
   }
}

//+------------------------------------------------------------------+
//| Normalized RSI: rescale(ema(rsi(src, n1), n2), 0, 100, 0, 1)    |
//+------------------------------------------------------------------+
void CalcNormalizedRSI(const double &src[], int n1, int n2, double &out[],
                       double &gain[], double &loss[],
                       double &gainRMA[], double &lossRMA[],
                       double &rawRSI[], double &emaRSI[],
                       int begin, int total)
{
   CalcRSI(src, n1, rawRSI, gain, loss, gainRMA, lossRMA, begin, total);
   CalcEMA(rawRSI, n2, emaRSI, begin, total);
   for(int i = begin; i < total; i++)
      out[i] = emaRSI[i] != LC_EMPTY ? ApplyRescale(emaRSI[i], 0, 100, 0, 1) : LC_EMPTY;
}

//+------------------------------------------------------------------+
//| Normalized CCI: normalize(ema(cci(src, n1), n2), 0, 1)          |
//+------------------------------------------------------------------+
void CalcNormalizedCCI(const double &src[], int n1, int n2, double &out[],
                       double &smaArr[], double &rawCCI[], double &emaCCI[],
                       NormState &normState, int begin, int total)
{
   CalcCCI(src, n1, rawCCI, smaArr, begin, total);
   CalcEMA(rawCCI, n2, emaCCI, begin, total);
   for(int i = begin; i < total; i++)
      out[i] = emaCCI[i] != LC_EMPTY ? ApplyNormalize(emaCCI[i], 0, 1, normState) : LC_EMPTY;
}

//+------------------------------------------------------------------+
//| Feature Engine - encapsulates per-feature state and computation  |
//+------------------------------------------------------------------+
enum ENUM_FEATURE_TYPE
{
   FEATURE_RSI = 0,
   FEATURE_WT  = 1,
   FEATURE_CCI = 2,
   FEATURE_ADX = 3
};

struct FeatureWork
{
   ENUM_FEATURE_TYPE type;
   int paramA;
   int paramB;
   double priceScale;   // 10^digits for ADX tick-quantization; 0 disables
   double output[];
   // RSI intermediates
   double rsiGain[], rsiLoss[], rsiGainRMA[], rsiLossRMA[], rsiRaw[], rsiEMA[];
   // WT intermediates
   double wtEma1[], wtAbsDev[], wtEma2[], wtCI[], wtWt1[], wtWt2[], wtRaw[];
   NormState wtNormState;
   // CCI intermediates
   double cciSMA[], cciRaw[], cciEMA[];
   NormState cciNormState;
   // ADX intermediates
   double adxTR[], adxDMPlus[], adxDMMinus[];
   double adxTRSmooth[], adxSmoothDMPlus[], adxSmoothDMMinus[];
   double adxDX[], adxRMA[];
};

void InitFeatureWork(FeatureWork &fw, ENUM_FEATURE_TYPE type, int paramA, int paramB,
                     double priceScale = 0.0)
{
   fw.type       = type;
   fw.paramA     = paramA;
   fw.paramB     = paramB;
   fw.priceScale = priceScale;
   if(type == FEATURE_WT)  InitNormState(fw.wtNormState);
   if(type == FEATURE_CCI) InitNormState(fw.cciNormState);
}

void ResizeFeatureWork(FeatureWork &fw, int total)
{
   ArrayResize(fw.output, total);
   switch(fw.type)
   {
      case FEATURE_RSI:
         ArrayResize(fw.rsiGain, total);    ArrayResize(fw.rsiLoss, total);
         ArrayResize(fw.rsiGainRMA, total); ArrayResize(fw.rsiLossRMA, total);
         ArrayResize(fw.rsiRaw, total);     ArrayResize(fw.rsiEMA, total);
         break;
      case FEATURE_WT:
         ArrayResize(fw.wtEma1, total);   ArrayResize(fw.wtAbsDev, total);
         ArrayResize(fw.wtEma2, total);   ArrayResize(fw.wtCI, total);
         ArrayResize(fw.wtWt1, total);    ArrayResize(fw.wtWt2, total);
         ArrayResize(fw.wtRaw, total);
         break;
      case FEATURE_CCI:
         ArrayResize(fw.cciSMA, total); ArrayResize(fw.cciRaw, total);
         ArrayResize(fw.cciEMA, total);
         break;
      case FEATURE_ADX:
         ArrayResize(fw.adxTR, total);          ArrayResize(fw.adxDMPlus, total);
         ArrayResize(fw.adxDMMinus, total);     ArrayResize(fw.adxTRSmooth, total);
         ArrayResize(fw.adxSmoothDMPlus, total); ArrayResize(fw.adxSmoothDMMinus, total);
         ArrayResize(fw.adxDX, total);          ArrayResize(fw.adxRMA, total);
         break;
   }
}

void CalcFeature(FeatureWork &fw,
                 const double &close[], const double &high[],
                 const double &low[], const double &hlc3[],
                 int begin, int total)
{
   switch(fw.type)
   {
      case FEATURE_RSI:
         CalcNormalizedRSI(close, fw.paramA, fw.paramB, fw.output,
                           fw.rsiGain, fw.rsiLoss, fw.rsiGainRMA, fw.rsiLossRMA,
                           fw.rsiRaw, fw.rsiEMA, begin, total);
         break;
      case FEATURE_WT:
         CalcWaveTrend(hlc3, fw.paramA, fw.paramB, fw.output,
                       fw.wtEma1, fw.wtAbsDev, fw.wtEma2,
                       fw.wtCI, fw.wtWt1, fw.wtWt2, fw.wtRaw,
                       fw.wtNormState, begin, total);
         break;
      case FEATURE_CCI:
         CalcNormalizedCCI(close, fw.paramA, fw.paramB, fw.output,
                           fw.cciSMA, fw.cciRaw, fw.cciEMA,
                           fw.cciNormState, begin, total);
         break;
      case FEATURE_ADX:
         CalcADX(high, low, close, fw.paramA, fw.output,
                 fw.adxTR, fw.adxDMPlus, fw.adxDMMinus,
                 fw.adxTRSmooth, fw.adxSmoothDMPlus, fw.adxSmoothDMMinus,
                 fw.adxDX, fw.adxRMA, begin, total, fw.priceScale);
         break;
   }
}

#endif // __LC_ML_FEATURES_MQH__
