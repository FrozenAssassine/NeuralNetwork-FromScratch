using NNFromScratch.Helper;
using System;

namespace NNFromScratch.Core.Layers
{
    public class InputLayer : BaseLayer
    {
        public InputLayer(int size)
        {
            this.Size = size;
        }

        public override void FeedForward()
        {
            //nothing to feed forward here
        }

        public override void Initialize(BaseLayer previousLayer)
        {
            LayerInitialisationHelper.InitializeLayer(this, previousLayer);
        }

        public override void InitializeCuda(int index)
        {
            CudaAccel.InitInputLayer(index, this.Size, this.Biases, this.Weights, this.NeuronValues, this.Errors, this.ActivationFunction);
        }

        public override void Load(BinaryReader br)
        {
            //nothing to load to input layer
        }

        public override void Save(BinaryWriter bw)
        {
            //nothing to save for input layer
        }

        public override void Summary()
        {
            Console.WriteLine($"Input Layer of {Size} Neurons and {Weights.Length} Weights");
        }

        public override void Train(float[] desiredValues, float learningRate)
        {
            //nothing to train here
        }
    }
}
