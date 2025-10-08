using NNFromScratch.Helper;

namespace NNFromScratch.Core;

internal class LossCalculator
{
    private float totalLossValue = 0;
    private int lossCount = 0;
    private NNModel nnModel;
    public LossCalculator(NNModel nn)
    {
        this.nnModel = nn;
    }
    public void Calculate(float[] desired)
    {
        float[] vals;
        if (nnModel.useCuda)
            vals = CudaAccel.GetOutputNeuronVals();
        else
            vals = nnModel.nn.allLayer[nnModel.nn.allLayer.Length - 1].NeuronValues;

        lossCount++;
        totalLossValue += LossHelper.MSE(vals, desired);
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
