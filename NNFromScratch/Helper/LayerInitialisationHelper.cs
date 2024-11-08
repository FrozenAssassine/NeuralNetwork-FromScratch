using NNFromScratch.Core.Layers;

namespace NNFromScratch.Helper;

internal class LayerInitialisationHelper
{
    public static void FillRandom(float[] biases, float[] weights)
    {
        //Maybe use "Xavier Initialization" ref: Finn Chat DC
        for (int i = 0; i < biases.Length; i++)
        {
            biases[i] = MathHelper.RandomBias();
        }

        if (weights == null)
            return;

        for (int i = 0; i < weights.Length; i++)
        {
            weights[i] = MathHelper.RandomWeight();
        }
    }

    public static void FillRandom(float[] array)
    {
        for (int i = 0; i < array.Length; i++)
        {
            array[i] = MathHelper.RandomWeight();
        }
    }
    
    public static void InitializeLayer(BaseLayer layer)
    {
        layer.Biases = new float[layer.Size];
        layer.NeuronValues = new float[layer.Size];
        layer.Errors = new float[layer.Size];

        //first layer does not have previousLayer:
        if(layer.PreviousLayer != null)
            layer.Weights = new float[layer.Size * layer.PreviousLayer.Size];

        FillRandom(layer.Biases, layer.Weights);
    }
}
