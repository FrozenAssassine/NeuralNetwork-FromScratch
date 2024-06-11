using NNFromScratch.Helper;
using System.Drawing;

namespace NNFromScratch.Core
{
    internal class Layer
    {
        public float[] Biases;
        public float[] NeuronValues;
        public float[] Errors;
        public float[,] Weight;
        public readonly int Size;
        public Layer PreviousLayer;
        public Layer followingLayer;
        public string Name;

        public Layer(int size, string name)
        {
            this.Size = size;
            this.Name = name;
        }

        public void Initialize(Layer previousLayer, Layer followingLayer)
        {
            this.Biases = new float[this.Size];
            this.NeuronValues = new float[this.Size];
            this.Errors = new float[this.Size];
            this.PreviousLayer = previousLayer;
            this.followingLayer = followingLayer;

            //store the weights between the current and previous layer or null if the current layer is the input layer
            if (previousLayer != null)
                this.Weight = new float[previousLayer.Size, this.Size];

            FillRandom();

            if (previousLayer != null)
                Console.WriteLine("\tConnected with " + previousLayer.Name + " with " + Weight.Length + " Weights");

            Console.WriteLine("Initialize layer " + Name + " with " + Size + " Neurons");
        }

        private void FillRandom()
        {
            //Maybe use "Xavier Initialization" ref: Finn Chat DC
            for (int i = 0; i < PreviousLayer.Size; i++)
            {
                Biases[i] = MathHelper.RandomBias();
                for (int j = 0; j < this.Size; j++)
                {
                    Weight[i, j] = MathHelper.RandomWeight();
                }
            }
        }

        public virtual void Train(float learningRate)
        {
            for(int i = 0; i<Errors.Length; i++)
            {
                double followingWeightedErrors = 0;
                for(int j = 0; j< followingLayer.NeuronValues.Length; j++)
                {
                    followingWeightedErrors += followingLayer.Errors[j] * followingLayer.Weight[j, i];
                }
                Errors[i] = MathHelper.SigmoidDerivative(NeuronValues[i]);

                for(int j = 0; j<PreviousLayer.NeuronValues.Length j++)
                {
                    Weight[i, j] += learningRate * Errors[i] * PreviousLayer.NeuronValues[j];
                }
                Biases[i] += learningRate * Errors[i];
            }
            PreviousLayer.Train(learningRate);
        }

        public virtual float[] FeedForward()
        {
            foreach (var hidden in hiddenLayers)
            {
                for (int i = 0; i < hidden.Size; i++)
                {
                    for (int j = 0; j < hidden.PreviousLayer.Size; j++)
                    {
                        hidden.NeuronValues[i] = MathHelper.Sigmoid(hidden.Weight[i * hidden.PreviousLayer.NeuronValues.Length + j] + hidden.Biases[i]);
                    }
                }
            }


            return outputLayer.NeuronValues;
        }
    }
}
