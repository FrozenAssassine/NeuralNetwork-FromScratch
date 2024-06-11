namespace NNFromScratch.Helper;

public class MathHelper
{
    private static Random random = new Random();
    public static float Sigmoid(float x)
    {
        return (1 / (1 + MathF.Exp(-x)));
    }

    public static float SigmoidDerivative(float x)
    {
        return x * (1 - x);
    }
    public static float RandomWeight()
    {
        return (float)random.NextDouble();
    }

    public static float RandomBias()
    {
        return (float)random.NextDouble();
    }

    public static int GetMaximumIndex(float[] items)
    {
        return Array.IndexOf(items, items.Max());
    }
}
