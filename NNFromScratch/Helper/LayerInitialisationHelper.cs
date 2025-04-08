using NNFromScratch.Core.Layers;

namespace NNFromScratch.Helper;

internal class LayerInitialisationHelper
{
    public static void FillRandom(float[] arr)
    {
        for (int i = 0; i < arr.Length; i++)
        {
            arr[i] = MathHelper.RandomBias();
        }
    }

    public static float[] XavierInitializeWeights(int fanIn, int fanOut, int totalWeights)
    {
        float limit = (float)Math.Sqrt(6.0 / (fanIn + fanOut));
        float[] weights = new float[totalWeights];

        for (int i = 0; i < totalWeights; i++)
        {
            weights[i] = MathHelper.RandomFloat(-limit, limit);
        }
        return weights;
    }

    public static void InitializeLayer(BaseLayer layer, int inputCount, int outputCount)
    {


        InitializeLayer(layer, inputCount, outputCount, layer.PreviousLayer == null ?  -1 : layer.Size * layer.PreviousLayer.Size);
    }

    public static void InitializeLayer(BaseLayer layer, int inputCount, int outputCount, int weightCount)
    {
        layer.Biases = new float[layer.Size];
        layer.NeuronValues = new float[layer.Size];
        layer.Errors = new float[layer.Size];

        //first layer does not have previousLayer:
        if (layer.PreviousLayer != null || weightCount != -1)
            layer.Weights = XavierInitializeWeights(inputCount, outputCount, weightCount);

        FillRandom(layer.Biases);
    }
}
