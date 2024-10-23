using NNFromScratch.Helper;

namespace NNFromScratch.Core;

internal class LossCalculator
{
    private float totalLossValue = 0;
    private int lossCount = 0;
    private NeuralNetwork nn;
    public LossCalculator(NeuralNetwork nn)
    {
        this.nn = nn;
    }
    public void Calculate(float[] desired)
    {
        lossCount++;
        totalLossValue += LossHelper.MSE(nn.allLayer[nn.allLayer.Length - 1].NeuronValues, desired);
    }

    public void PrintLoss()
    {
        Console.WriteLine($"Loss {MakeLoss()}");
    }
    public float MakeLoss()
    {
        return MathF.Round(totalLossValue / lossCount, 4);
    }
    public void NextEpoch()
    {
        lossCount = 0;
        totalLossValue = 0.0f;
    }
}
