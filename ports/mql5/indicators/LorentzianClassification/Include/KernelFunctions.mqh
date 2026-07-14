//+------------------------------------------------------------------+
//|                                           KernelFunctions.mqh    |
//| Nadaraya-Watson kernel regression.                                |
//+------------------------------------------------------------------+
//
// ============================================
// ==== Nadaraya-Watson Kernel Regression  ====
// ============================================
//
// Nadaraya-Watson kernel regression is a non-parametric method used to
// estimate the conditional expectation of a random variable. It applies
// kernel-weighted averaging so that recent observations carry more weight
// than distant ones, producing a smoothed "expected price" curve without
// the hard edges of a simple moving average.
//
// This indicator uses kernel regression in two roles:
//   (1) As a visual overlay on the chart (the green/red smoothed curve).
//   (2) As an optional prediction filter: the kernel's direction and
//       crossover behaviour gates ML entry signals.
//
// Two kernels are provided:
//   - Rational Quadratic (primary): an infinite sum of Gaussian kernels of
//     different length scales. The relativeWeight parameter blends scales:
//     as it approaches zero, longer time frames exert more influence on the
//     estimation; as it approaches infinity, the behaviour of the Rational
//     Quadratic kernel becomes identical to the Gaussian kernel.
//   - Gaussian (secondary): a weighted average using a Radial Basis
//     Function (RBF). Used here with a reduced lookback (h - lag) to
//     produce a faster line whose crossovers with the primary kernel give
//     earlier entry/exit alerts.
//
// Both implementations are non-repainting: they only use bars at or before
// barIndex. startAtBar controls how tightly the fit follows the most
// recent price -- smaller values produce a tighter fit, larger values
// produce a looser, smoother fit. Recommended range: 2-25.
//
#ifndef __LC_KERNEL_FUNCTIONS_MQH__
#define __LC_KERNEL_FUNCTIONS_MQH__

//+------------------------------------------------------------------+
//| Rational Quadratic Kernel                                        |
//| An infinite sum of Gaussian kernels of different length scales.  |
//| src[] is forward-indexed (0=oldest). barIndex is current bar.    |
//|                                                                  |
//| Parameters:                                                      |
//|   lookback       - number of bars used for the estimation (h).   |
//|                    A sliding window over the most recent bars.   |
//|                    Recommended range: 3-50.                      |
//|   relativeWeight - relative weighting of time frames.            |
//|                    ->0   : longer time frames dominate.          |
//|                    ->inf : behaves like the Gaussian kernel.     |
//|                    Recommended range: 0.25-25.                   |
//|   startAtBar     - bar offset at which regression begins.        |
//|                    Smaller = tighter fit; larger = looser fit.   |
//|                    Recommended range: 2-25.                      |
//+------------------------------------------------------------------+
double KernelRationalQuadratic(const double &src[], int barIndex,
                               int lookback, double relativeWeight, int startAtBar)
{
   double currentWeight    = 0;
   double cumulativeWeight = 0;
   // Walk i = 0 .. 1 + startAtBar, clamped to available history.
   int limit = 1 + startAtBar;
   for(int i = 0; i <= limit && i <= barIndex; i++)
   {
      double y = src[barIndex - i];
      double denom = MathPow((double)lookback, 2) * 2.0 * relativeWeight;
      if(denom == 0) denom = 1e-10;
      double w = MathPow(1.0 + (MathPow((double)i, 2) / denom),
                  -relativeWeight);
      currentWeight    += y * w;
      cumulativeWeight += w;
   }
   return cumulativeWeight > 0 ? currentWeight / cumulativeWeight : src[barIndex];
}

//+------------------------------------------------------------------+
//| Gaussian Kernel                                                  |
//| Weighted average using a Radial Basis Function (RBF). Produces   |
//| a smoother, slightly lagging companion to the Rational Quadratic |
//| estimate. Used in this indicator with lookback = h - lag so that |
//| its crossovers with the primary kernel drive the "smoothing"     |
//| alert/filter variants. Lower lag values result in earlier        |
//| crossovers. Recommended lag range: 1-2.                          |
//+------------------------------------------------------------------+
double KernelGaussian(const double &src[], int barIndex,
                      int lookback, int startAtBar)
{
   double currentWeight    = 0;
   double cumulativeWeight = 0;
   int limit = 1 + startAtBar;
   for(int i = 0; i <= limit && i <= barIndex; i++)
   {
      double y = src[barIndex - i];
      double gDenom = 2.0 * MathPow((double)lookback, 2);
      if(gDenom == 0) gDenom = 1e-10;
      double w = MathExp(-MathPow((double)i, 2) / gDenom);
      currentWeight    += y * w;
      cumulativeWeight += w;
   }
   return cumulativeWeight > 0 ? currentWeight / cumulativeWeight : src[barIndex];
}

#endif // __LC_KERNEL_FUNCTIONS_MQH__
