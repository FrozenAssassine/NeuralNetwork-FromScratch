using System;

namespace NNFromScratch.Core.Layers
{
    public class InputLayer : NeuronLayer
    {
        public InputLayer(int size) : base(size, ActivationType.Sigmoid) { }
    }
}
