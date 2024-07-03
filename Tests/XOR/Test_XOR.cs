using NNFromScratch.Core;
using NNFromScratch.Core.Layers;

namespace Tests.XOR;

internal class Test_XOR
{
    public static void Run()
    {
        var nnmodel = NetworkBuilder.Create()
            .Stack(new InputLayer(2))
            .Stack(new DenseLayer(4, ActivationType.Sigmoid))
            .Stack(new OutputLayer(2, ActivationType.Sigmoid))
            .Build();

        nnmodel.Summary();

        float[][] inputs = new float[][] { new float[] { 0, 0 }, new float[] { 0, 1 }, new float[] { 1, 0 }, new float[] { 1, 1 } };
        float[][] desired = new float[][] { new float[] { 0 }, new float[] { 1 }, new float[] { 1 }, new float[] { 0 } };
        nnmodel.Train(inputs, desired, 15900, 0.01f);

        var predict = nnmodel.Predict(new float[] { 0, 0 });
        foreach (var pred in predict)
        {
            Console.WriteLine(pred);
        }
    }
}