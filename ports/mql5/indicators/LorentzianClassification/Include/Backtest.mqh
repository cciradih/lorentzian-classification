//+------------------------------------------------------------------+
//|                                                   Backtest.mqh   |
//| Trade stats accumulation and on-chart table display.             |
//+------------------------------------------------------------------+
//
// =====================
// ==== Backtesting ====
// =====================
//
// The stats computed here can be used to display real-time trade stats. In
// the context of Feature Engineering this is a useful mechanism for
// obtaining real-time feedback. It does NOT replace the need to properly
// backtest. The trade-stats section is intended for calibration purposes
// only -- i.e. comparing relative performance of different Feature
// Engineering choices, NOT as a performance claim.
//
// Signal semantics (consumed by UpdateBacktest, one call per bar):
//   startLong  : open long at this bar's marketPrice.
//   endLong    : close long, realize delta = marketPrice - startLongPrice.
//   startShort : open short at this bar's marketPrice.
//   endShort   : close short, realize delta = startShortPrice - marketPrice.
// The same information is surfaced to chart consumers via the BuyBuf /
// SellBuf / ExitBuyBuf / ExitSellBuf indicator outputs.
//
// Definitions:
//   - Winrate  = cumWins / cumTrades.
//   - W/L Ratio = cumWins / cumLosses.
//   - Early Signal Flips ("Stop-Loss"): in this context a "stop-loss" is
//     defined as an instance where the ML Signal prematurely flips
//     directions before an exit signal can be generated. High values can
//     indicate choppy (ranging) market conditions.
//
// Entry price modeling:
//   - Default: marketPrice = (high + low + 2*open) / 4 -- a proxy for the
//     mid-bar fill a discretionary trader might achieve.
//   - useWorstCase=true: marketPrice = src (typically close). This assumes
//     the user waits for the bar to close before acting, which avoids the
//     effects of intrabar repainting and produces a conservative estimate.
//     On larger timeframes this can mean entering after a large move has
//     already occurred. Leaving this option disabled is generally better
//     for those that use this indicator as a source of confluence and
//     prefer estimates that demonstrate discretionary mid-bar entries.
//     Leaving this option enabled may be more consistent with traditional
//     backtesting results.
//
#ifndef __LC_BACKTEST_MQH__
#define __LC_BACKTEST_MQH__

struct BacktestState
{
   double startLongPrice;
   double startShortPrice;
   double totalLongProfit;
   double totalShortProfit;
   double grossProfit;
   double grossLoss;
   int    wins;
   int    losses;
   int    tradeCount;
   int    earlyFlips;
   // Cumulative
   double cumWins;
   double cumLosses;
   double cumTrades;
   double cumEarlyFlips;
   double cumGrossProfit;
   double cumGrossLoss;
};

void InitBacktest(BacktestState &bs)
{
   bs.startLongPrice  = 0;
   bs.startShortPrice = 0;
   bs.totalLongProfit = 0;
   bs.totalShortProfit = 0;
   bs.grossProfit = 0;
   bs.grossLoss   = 0;
   bs.wins = 0; bs.losses = 0; bs.tradeCount = 0; bs.earlyFlips = 0;
   bs.cumWins = 0; bs.cumLosses = 0; bs.cumTrades = 0;
   bs.cumEarlyFlips = 0; bs.cumGrossProfit = 0; bs.cumGrossLoss = 0;
}

//+------------------------------------------------------------------+
//| Update backtest stats for one bar                                |
//+------------------------------------------------------------------+
void UpdateBacktest(BacktestState &bs,
                    double high, double low, double open, double src,
                    bool startLong, bool endLong,
                    bool startShort, bool endShort,
                    bool isEarlyFlip, bool useWorstCase)
{
   double marketPrice = useWorstCase ? src : (high + low + open + open) / 4.0;

   // Reset per-bar accumulators
   bs.tradeCount = 0; bs.wins = 0; bs.losses = 0;
   bs.earlyFlips = 0; bs.grossProfit = 0; bs.grossLoss = 0;
   bs.totalLongProfit = 0; bs.totalShortProfit = 0;

   if(startLong)
   {
      bs.startShortPrice = 0;
      bs.earlyFlips  = isEarlyFlip ? 1 : 0;
      bs.startLongPrice = marketPrice;
      bs.tradeCount = 1;
   }
   if(endLong)
   {
      double delta = marketPrice - bs.startLongPrice;
      bs.wins   = delta > 0 ? 1 : 0;
      bs.losses = delta < 0 ? 1 : 0;
      bs.totalLongProfit = delta;
      if(delta > 0) bs.grossProfit = delta;
      else          bs.grossLoss  = MathAbs(delta);
   }
   if(startShort)
   {
      bs.startLongPrice = 0;
      bs.startShortPrice = marketPrice;
      bs.tradeCount = 1;
   }
   if(endShort)
   {
      bs.earlyFlips = isEarlyFlip ? 1 : 0;
      double delta = bs.startShortPrice - marketPrice;
      bs.wins   = delta > 0 ? 1 : 0;
      bs.losses = delta < 0 ? 1 : 0;
      bs.totalShortProfit = delta;
      if(delta > 0) bs.grossProfit = delta;
      else          bs.grossLoss  = MathAbs(delta);
   }

   // Accumulate
   bs.cumWins       += bs.wins;
   bs.cumLosses     += bs.losses;
   bs.cumTrades     += bs.wins + bs.losses;
   bs.cumEarlyFlips += bs.earlyFlips;
   bs.cumGrossProfit += bs.grossProfit;
   bs.cumGrossLoss   += bs.grossLoss;
}

//+------------------------------------------------------------------+
//| Draw stats table as chart objects in top-right corner             |
//+------------------------------------------------------------------+
void DrawStatsTable(const BacktestState &bs)
{
   string prefix = "LC_Stats_";
   // (No background rectangle; stats render as plain text on the chart.)

   double winRate = bs.cumTrades > 0 ? bs.cumWins / bs.cumTrades * 100.0 : 0;
   double wlRatio = bs.cumLosses > 0 ? bs.cumWins / bs.cumLosses : 0;

   // Helper lambda-like approach: create labels at fixed positions
   int y = 25;
   int lineH = 18;
   CreateStatsLabel(prefix + "title", 15, y, "Trade Stats", clrWhite, 10, true);
   y += lineH + 4;
   CreateStatsLabel(prefix + "wr_l", 15, y, "Win Rate",  C'180,180,190', 9, false);
   CreateStatsLabel(prefix + "wr_v", 155, y, DoubleToString(winRate, 1) + "%", clrWhite, 9, false);
   y += lineH;
   CreateStatsLabel(prefix + "tr_l", 15, y, "Trades",    C'180,180,190', 9, false);
   CreateStatsLabel(prefix + "tr_v", 155, y,
                    IntegerToString((int)bs.cumTrades) + " (" +
                    IntegerToString((int)bs.cumWins) + "|" +
                    IntegerToString((int)bs.cumLosses) + ")", clrWhite, 9, false);
   y += lineH;
   CreateStatsLabel(prefix + "wl_l", 15, y, "W/L Ratio", C'180,180,190', 9, false);
   CreateStatsLabel(prefix + "wl_v", 155, y, DoubleToString(wlRatio, 2), clrWhite, 9, false);
   y += lineH;
   CreateStatsLabel(prefix + "ef_l", 15, y, "Early Flips", C'180,180,190', 9, false);
   CreateStatsLabel(prefix + "ef_v", 155, y, IntegerToString((int)bs.cumEarlyFlips), clrWhite, 9, false);
}

void CreateStatsLabel(string name, int x, int y, string text,
                      color clr, int fontSize, bool isBold)
{
   ObjectCreate(0, name, OBJ_LABEL, 0, 0, 0);
   ObjectSetInteger(0, name, OBJPROP_CORNER, CORNER_RIGHT_UPPER);
   ObjectSetInteger(0, name, OBJPROP_XDISTANCE, 290 - x);
   ObjectSetInteger(0, name, OBJPROP_YDISTANCE, y);
   ObjectSetString(0, name, OBJPROP_TEXT, text);
   ObjectSetInteger(0, name, OBJPROP_COLOR, clr);
   ObjectSetInteger(0, name, OBJPROP_FONTSIZE, fontSize);
   ObjectSetString(0, name, OBJPROP_FONT, isBold ? "Arial Bold" : "Arial");
}

void DeleteStatsTable()
{
   ObjectsDeleteAll(0, "LC_Stats_");
}

#endif // __LC_BACKTEST_MQH__
