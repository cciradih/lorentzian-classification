//+------------------------------------------------------------------+
//|                                                        ANN.mqh   |
//| Greedy Approximate Nearest Neighbors classifier with Lorentzian  |
//| distance.                                                         |
//+------------------------------------------------------------------+
//
// =========================
// ====  Core ML Logic  ====
// =========================
//
// Approximate Nearest Neighbors Search with Lorentzian Distance:
// A novel variation of the Nearest Neighbors (NN) search algorithm that
// ensures a chronologically uniform distribution of neighbors.
//
// In a traditional KNN-based approach, we would iterate through the entire
// dataset and calculate the distance between the current bar and every
// other bar in the dataset, then sort the distances in ascending order. We
// would then take the first k bars and use their labels to determine the
// label of the current bar.
//
// There are several problems with this traditional KNN approach in the
// context of real-time calculations involving time-series data:
//   - It is computationally expensive to iterate through the entire dataset
//     and calculate the distance between every historical bar and the
//     current bar.
//   - Market time-series data is often non-stationary, meaning that the
//     statistical properties of the data change slightly over time.
//   - It is possible that the nearest neighbors are not the most
//     informative ones, and the KNN algorithm may return poor results if
//     the nearest neighbors are not representative of the majority of the
//     data.
//
// Previously, some KNN implementations attempted to address some of
// these issues by:
//   - Using a modified KNN algorithm based on consecutive furthest neighbors
//     to find a set of approximate "nearest" neighbors.
//   - Using a sliding window approach to only calculate the distance
//     between the current bar and the most recent n bars in the dataset.
//
// Of these two approaches, the latter is inherently limited by the fact
// that it only considers the most recent bars in the overall dataset.
//
// The former approach has more potential to leverage historical price
// action, but is limited by:
//   - The possibility of a sudden "max" value throwing off the estimation.
//   - The possibility of selecting a set of approximate neighbors that are
//     not representative of the majority of the data by oversampling
//     values that are not chronologically distinct enough from one another.
//   - The possibility of selecting too many "far" neighbors, which may
//     result in a poor estimation of price action.
//
// To address these issues, a novel Approximate Nearest Neighbors (ANN)
// algorithm is used in this indicator.
//
// In the below ANN algorithm:
//   1. The algorithm iterates through the dataset in chronological order,
//      using the modulo operator to only perform calculations every 4 bars.
//      This serves the dual purpose of reducing the computational overhead
//      of the algorithm and ensuring a minimum chronological spacing
//      between the neighbors of at least 4 bars.
//   2. A list of the k-similar neighbors is simultaneously maintained in
//      both a predictions array and corresponding distances array.
//   3. When the size of the predictions array exceeds the desired number of
//      nearest neighbors specified in settings.neighborsCount, the
//      algorithm removes the first neighbor from the predictions array and
//      the corresponding distance array.
//   4. The lastDistance variable is overriden to be a distance in the lower
//      25% of the array. This step helps to boost overall accuracy by
//      ensuring subsequent newly added distance values increase at a slower
//      rate.
//   5. Lorentzian distance is used as a distance metric in order to
//      minimize the effect of outliers and take into account the warping of
//      "price-time" due to proximity to significant economic events.
//
#ifndef __ANN_MQH__
#define __ANN_MQH__

// LC_EMPTY sentinel for warmup values is defined in MLFeatures.mqh.
#include "MLFeatures.mqh"

// =====================================================================
// ANN state: feature history + classification output
// =====================================================================
struct ANNState
{
   // Feature history -- forward-indexed (0 = oldest bar, N-1 = most recent)
   double f1[];
   double f2[];
   double f3[];
   double f4[];
   double f5[];
   // Training labels: +1 long, -1 short, 0 neutral
   int    trainLabels[];
   // Number of accumulated bars
   int    dataSize;
   // Greedy-queue state -- persists across bars so the neighbor pool
   // accumulates across the entire history, not per-bar.
   double distances[];
   double predictions[];
   int    neighborBarIdx[];
   // Classification output (sum of neighbor votes for the current bar)
   int    prediction;
};

//+------------------------------------------------------------------+
//| Zero-initialize all state                                        |
//+------------------------------------------------------------------+
void InitANN(ANNState &st)
{
   ArrayResize(st.f1, 0);
   ArrayResize(st.f2, 0);
   ArrayResize(st.f3, 0);
   ArrayResize(st.f4, 0);
   ArrayResize(st.f5, 0);
   ArrayResize(st.trainLabels, 0);
   ArrayResize(st.distances,      0);
   ArrayResize(st.predictions,    0);
   ArrayResize(st.neighborBarIdx, 0);
   st.dataSize  = 0;
   st.prediction = 0;
}

//+------------------------------------------------------------------+
//| Append one bar's features and label to history                   |
//+------------------------------------------------------------------+
void ANNPushBar(ANNState &st, double v1, double v2, double v3,
                double v4, double v5, int label)
{
   int sz = st.dataSize;
   ArrayResize(st.f1, sz + 1); st.f1[sz] = v1;
   ArrayResize(st.f2, sz + 1); st.f2[sz] = v2;
   ArrayResize(st.f3, sz + 1); st.f3[sz] = v3;
   ArrayResize(st.f4, sz + 1); st.f4[sz] = v4;
   ArrayResize(st.f5, sz + 1); st.f5[sz] = v5;
   ArrayResize(st.trainLabels, sz + 1); st.trainLabels[sz] = label;
   st.dataSize = sz + 1;
}

//+------------------------------------------------------------------+
//| Lorentzian distance: sum of log(1 + |delta|) over active features|
//|                                                                  |
//| If any active feature (current or history) is LC_EMPTY we return |
//| -DBL_MAX so the caller's `d >= lastDistance` test fails and the  |
//| warmup candidate is silently skipped. This prevents NaN / -inf   |
//| from polluting the distance queue.                               |
//+------------------------------------------------------------------+
double GetLorentzianDistance(const ANNState &st, int idx, int featureCount,
                             double f1, double f2, double f3, double f4, double f5)
{
   if(featureCount >= 1 && (f1 == LC_EMPTY || st.f1[idx] == LC_EMPTY)) return -DBL_MAX;
   if(featureCount >= 2 && (f2 == LC_EMPTY || st.f2[idx] == LC_EMPTY)) return -DBL_MAX;
   if(featureCount >= 3 && (f3 == LC_EMPTY || st.f3[idx] == LC_EMPTY)) return -DBL_MAX;
   if(featureCount >= 4 && (f4 == LC_EMPTY || st.f4[idx] == LC_EMPTY)) return -DBL_MAX;
   if(featureCount >= 5 && (f5 == LC_EMPTY || st.f5[idx] == LC_EMPTY)) return -DBL_MAX;

   double d = 0;
   if(featureCount >= 1) d += MathLog(1.0 + MathAbs(f1 - st.f1[idx]));
   if(featureCount >= 2) d += MathLog(1.0 + MathAbs(f2 - st.f2[idx]));
   if(featureCount >= 3) d += MathLog(1.0 + MathAbs(f3 - st.f3[idx]));
   if(featureCount >= 4) d += MathLog(1.0 + MathAbs(f4 - st.f4[idx]));
   if(featureCount >= 5) d += MathLog(1.0 + MathAbs(f5 - st.f5[idx]));
   return d;
}

//+------------------------------------------------------------------+
//| Core greedy ANN loop.                                             |
//|                                                                   |
//| Algorithm:                                                        |
//|   lastDistance = -1  (reset per bar; NOT persistent)              |
//|   sizeLoop = min(maxBarsBack - 1, dataSize - 1)                   |
//|   for i = 0..sizeLoop:                                            |
//|      d = distance(currentFeatures, history[i])                    |
//|      if d >= lastDistance AND (i % 4) != 0:                       |
//|          lastDistance = d                                         |
//|          push (d, round(trainLabels[i])) to st.distances/preds    |
//|          if st.predictions size > K:                              |
//|              lastDistance = st.distances[round(K * 3 / 4)]        |
//|              shift front off both queues                          |
//|   prediction = sum(st.predictions)                                |
//|                                                                   |
//| st.distances and st.predictions persist across bars so the        |
//| neighbor pool accumulates over the full history.                  |
//+------------------------------------------------------------------+
void RunANN(ANNState &st, int neighborsCount, int featureCount, int maxBarsBack,
            double f1, double f2, double f3, double f4, double f5,
            bool includeFullHistory, int lastBarIndex)
{
   st.prediction = 0;
   if(st.dataSize <= 0) return;

   int sizeLoop = MathMin(maxBarsBack - 1, st.dataSize - 1);

   // startIndex = includeFullHistory ? 0 : maxBarsBackIndex
   // maxBarsBackIndex = last_bar_index - maxBarsBack (constant across all bars)
   int maxBarsBackIndex = (lastBarIndex >= maxBarsBack) ? lastBarIndex - maxBarsBack : 0;
   int startIdx = includeFullHistory ? 0 : maxBarsBackIndex;
   int endIdx   = sizeLoop;
   int step     = (startIdx > endIdx) ? -1 : 1;

   double lastDistance = -1.0;

   for(int i = startIdx; (step > 0) ? (i <= endIdx) : (i >= endIdx); i += step)
   {
      double d = GetLorentzianDistance(st, i, featureCount, f1, f2, f3, f4, f5);

      // (i % 4) != 0 enforces a minimum 4-bar chronological spacing
      // between neighbors while also reducing compute.
      if(d >= lastDistance && (i % 4) != 0)
      {
         lastDistance = d;

         int qSize = ArraySize(st.distances);
         ArrayResize(st.distances,      qSize + 1); st.distances[qSize]      = d;
         ArrayResize(st.predictions,    qSize + 1); st.predictions[qSize]    = MathRound((double)st.trainLabels[i]);
         ArrayResize(st.neighborBarIdx, qSize + 1); st.neighborBarIdx[qSize] = i;

         if(ArraySize(st.predictions) > neighborsCount)
         {
            // Raise threshold to the distance at the 75th percentile index
            int thresholdIdx = (int)MathRound((double)neighborsCount * 3.0 / 4.0);
            lastDistance = st.distances[thresholdIdx];

            // Shift: remove first element of all queues
            int newSz = ArraySize(st.distances) - 1;
            for(int j = 0; j < newSz; j++)
            {
               st.distances[j]      = st.distances[j + 1];
               st.predictions[j]    = st.predictions[j + 1];
               st.neighborBarIdx[j] = st.neighborBarIdx[j + 1];
            }
            ArrayResize(st.distances,      newSz);
            ArrayResize(st.predictions,    newSz);
            ArrayResize(st.neighborBarIdx, newSz);
         }
      }
   }

   // Simple vote sum -- no tie-breaking (unlike TDANN)
   int predSize = ArraySize(st.predictions);
   for(int i = 0; i < predSize; i++)
      st.prediction += (int)st.predictions[i];
}

#endif // __ANN_MQH__
