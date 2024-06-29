using NNFromScratch.Core.ActivationFunctions;

namespace NNFromScratch.Core.Layers
{
    public class InputLayer : NeuronLayer
    {
        public InputLayer(int size, IActivationFunction activation) : base(size, activation) { }
    }
}
