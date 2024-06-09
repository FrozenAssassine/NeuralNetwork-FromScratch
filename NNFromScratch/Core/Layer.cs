using NNFromScratch.Helper;
using System.Drawing;

namespace NNFromScratch.Core
{
    internal class Layer
    {
        public float[] Biases;
        public float[] NeuronValues;
        public float[] Errors;
        public float[] Weight;
        public readonly int Size;
        public Layer PreviousLayer;
        public string Name;

        public Layer(int size, string name)
        {
            this.Size = size;
            this.Name = name;
        }

        public void Initialize(Layer previousLayer)
        {
            this.Biases = new float[this.Size];
            this.NeuronValues = new float[this.Size];
            this.Errors = new float[this.Size];
            this.PreviousLayer = previousLayer;

            //store the weights between the current and previous layer or null if the current layer is the input layer
            if (previousLayer != null)
                this.Weight = new float[previousLayer.Size * this.Size];

            FillRandom();

            if (previousLayer != null)
                Console.WriteLine("\tConnected with " + previousLayer.Name + " with " + Weight.Length + " Weights");

            Console.WriteLine("Initialize layer " + Name + " with " + Size + " Neurons");
        }

        private void FillRandom()
        {
            //Maybe use "Xavier Initialization" ref: Finn Chat DC
            for (int i = 0; i<Size; i++)
            {
                Biases[i] = MathHelper.RandomBias();
                if(Weight != null)
                    Weight[i] = MathHelper.RandomWeight();
            }
        }
    }
}
