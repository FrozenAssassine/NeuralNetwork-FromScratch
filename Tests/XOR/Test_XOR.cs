using NNFromScratch.Core;

namespace Tests.XOR;

internal class Test_XOR
{
    public static void Run()
    {
        NNModel nnmodel = new NNModel(new Layer[]
        {
                new Layer(2),
                new Layer(4),
                new Layer(1),
        });

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