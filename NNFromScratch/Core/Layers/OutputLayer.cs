
using NNFromScratch.Core.ActivationFunctions;

namespace NNFromScratch.Core.Layers;

public class OutputLayer : NeuronLayer
{
    public OutputLayer(int size, IActivationFunction activation) : base(size, activation)
    {
    }
}
