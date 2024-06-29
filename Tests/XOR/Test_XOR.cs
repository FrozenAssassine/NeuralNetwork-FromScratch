using NNFromScratch.Core;
using NNFromScratch.Core.ActivationFunctions;
using NNFromScratch.Core.Layers;

namespace Tests.XOR;

internal class Test_XOR
{
    public static void Run()
    {
        var activationFunction = new SigmoidActivation();
        var nnmodel = NetworkBuilder.Create()
            .Stack(new InputLayer(2, activationFunction))
            .Stack(new NeuronLayer(4, activationFunction))
            .Stack(new OutputLayer(2, activationFunction))
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