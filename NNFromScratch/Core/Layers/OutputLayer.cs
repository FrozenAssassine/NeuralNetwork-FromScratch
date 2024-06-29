
namespace NNFromScratch.Core.Layers;

public class OutputLayer : NeuronLayer
{
    public OutputLayer(int size, ActivationType activation = ActivationType.Sigmoid) : base(size, activation)
    {
    }
}
