using NNFromScratch.Core.ActivationFunctions;
using NNFromScratch.Core.Layers;

namespace NNFromScratch.Core
{
    public class NetworkBuilder
    {
        private List<NeuronLayer> layers = new();
        public static NetworkBuilder Create()
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
