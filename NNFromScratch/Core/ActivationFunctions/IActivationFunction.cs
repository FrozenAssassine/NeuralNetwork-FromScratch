namespace NNFromScratch.Core.ActivationFunctions;

public interface IActivationFunction
{
    float Calculate(float x);
    float CalculateDeriv(float x);
}
