namespace NNFromScratch.Helper;

public class MathHelper
{
    private static Random random = new Random();
    public static float RandomWeight()
    {
        return (random.NextSingle() * 2) - 1;
    }

    public static float RandomBias()
    {
        return (random.NextSingle() * 2) - 1;
    }

    public static int GetMaximumIndex(float[] items)
    {
        return Array.IndexOf(items, items.Max());
    }

    public static (int index, float value) GetMaxValIndex(float[] items)
    {
        int index = Array.IndexOf(items, items.Max());

        return (index, items[index]);
    }


    public static float RandomFloat(double min, double max)
    {
        return (float)(min + random.NextDouble() * (max - min));
    }

    public static double[,] GetPortion(double[] data, int originalWidth, int width, int height, int startX, int startY)
    {
        var result = new double[height, width];
        for (var y = 0; y < height; y++)
        {
            for (var x = 0; x < width; x++)
            {
                int index = (startY + y) * originalWidth + (startX + x);
                result[y, x] = data[index];
            }
        }

        return result;
    }
}
