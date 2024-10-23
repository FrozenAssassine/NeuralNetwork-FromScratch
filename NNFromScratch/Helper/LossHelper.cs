using NNFromScratch.Core;

namespace NNFromScratch.Helper;

internal class LossHelper
{
    public static float MSE(float[] predicted, float[] desired)
    {
        float sum = 0.0f;
        for (int i = 0; i < predicted.Length; i++)
        {
            float err = predicted[i] - desired[i];
            sum += err * err;
        }
        return sum / predicted.Length;
    }
}
