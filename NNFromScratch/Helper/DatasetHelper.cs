namespace NNFromScratch.Helper;

public class DatasetHelper
{
    public static void PrintItems(float[] items, int count)
    {
        Console.WriteLine(string.Join(", ", items.Take(count)));
    }
}
