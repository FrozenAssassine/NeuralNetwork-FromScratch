namespace NNFromScratch.Core.ActivationFunctions;

internal class SigmoidActivation : IActivationFunction
{
    public float Calculate(float x)
    {
        return 1.0f / (1.0f + MathF.Exp(-x));
    }
    
    public float CalculateDeriv(float x)
    {
        return x / (1.0f - x);
    }
}
