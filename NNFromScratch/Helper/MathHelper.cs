using NNFromScratch.Core;

namespace NNFromScratch.Helper;

internal class MathHelper
{
    private static Random random = new Random();
    /*public static double DotProduct(Neuron[] inputs)
    {
        double res = 0;
        for (int i = 0; i < inputs.Length; i++)
        {
            res += inputs[i].value * WeightHelper.AllWeights[inputs[i].weightIndex];
        }
        return res;
    }*/
    public static float Sigmoid(float x)
    {
        return (float)(1 / (1 + Math.Exp(-x)));
    }

    public static float SigmoidDerivative(float x)
    {
        return x * (1 - x);
    }

    public static float MSE_Loss(float[] y_true, float[] y_pred)
    {
        float sum = 0;
        for (int i = 0; i < y_true.Length; i++)
        {
            float diff = y_true[i] - y_pred[i];
            sum += diff * diff;
        }

        //average:
        return sum / y_true.Length;
    }

    public static float RandomWeight()
    {
        return (float)random.NextDouble();
    }

    public static float RandomBias()
    {
        return (float)random.NextDouble();
    }
}
