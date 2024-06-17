using NNFromScratch.Core.ActivationFunctions;
using NNFromScratch.Core.Layers;

namespace NNFromScratch.Core
{
    internal class NetworkBuilder
    {
        private List<NeuronLayer> layers = new();
        public NetworkBuilder Create()
        {
            return new NetworkBuilder();
        }

        public NetworkBuilder Stack(NeuronLayer layer)
        {
            layers.Add(layer);
            return this;
        }

        public NNModel Build(bool useCuda = true)
        {
            return new NNModel(layers.ToArray<NeuronLayer>(), useCuda);
        }
    }
}
