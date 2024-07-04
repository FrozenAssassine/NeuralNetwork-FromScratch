namespace NNFromScratch.Helper;

public class MathHelper
{
    private static Random random = new Random();

    public static float RandomFloat1_1()
    {
        return (random.NextSingle() * 2) - 1;
    }
    public static float RandomFloat0_1()
    {
        return random.NextSingle();
    }

    public static int GetMaximumIndex(float[] items)
    {
        return Array.IndexOf(items, items.Max());
    }
}
