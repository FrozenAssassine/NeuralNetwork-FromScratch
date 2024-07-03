using NNFromScratch.Helper;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNFromScratch.Core.Layers
{
    public abstract class BaseLayer
    {
        public float[] Biases;
        public float[] NeuronValues;
        public float[] Errors;
        public float[] Weights;
        public int Size;
        public BaseLayer PreviousLayer;
        public BaseLayer NextLayer;
        public ActivationType ActivationFunction;

        public abstract void FeedForward();

        public abstract void Train(float[] desiredValues, float learningRate);

        public abstract void Summary();

        public abstract void Save(BinaryWriter bw);
        public abstract void Load(BinaryReader br);
            
        public abstract void Initialize(BaseLayer previousLayer);

        public abstract void InitializeCuda(int index);
    };
}
