namespace NNFromScratch.Helper;

internal class ArrayHelper
{
    public static (float[] arr, int samples, int features) Flatten(float[][] inputArray)
    {
        int samples = inputArray.Length;
        int features = inputArray[0].Length;

        float[] flatArr = new float[samples * features];
        for (int i = 0; i < samples; i++)
            Buffer.BlockCopy(inputArray[i], 0, flatArr, i * features * sizeof(float), features * sizeof(float));
        return (flatArr, samples, features);
    }
}
