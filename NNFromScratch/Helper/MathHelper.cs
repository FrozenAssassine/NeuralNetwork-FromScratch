namespace NNFromScratch.Helper;

internal class MathHelper
{
    private static Random random = new Random();
    public static double DotProduct(double[] inputs, double[] weights)
    {
        double res = 0;
        for (int i = 0; i < inputs.Length; i++)
        {
            res += inputs[i] * weights[i];
        }
        return res;
    }
    public static double Sigmoid(double x)
    {
        return 1 / (1 + Math.Exp(-x));
    }

    public static double MSE_Loss(double[] y_true, double[] y_pred)
    {
        double sum = 0;
        for (int i = 0; i < y_true.Length; i++)
        {
            double diff = y_true[i] - y_pred[i];
            sum += diff * diff;
        }

        //average:
        return sum / y_true.Length;
    }

    public static double RandomWeight()
    {
        return random.NextDouble();
    }

    public static double RandomBias()
    {
        return random.NextDouble();
    }
}
