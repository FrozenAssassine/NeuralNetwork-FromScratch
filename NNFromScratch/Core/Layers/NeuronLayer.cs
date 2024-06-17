using NNFromScratch.Core.ActivationFunctions;
using NNFromScratch.Helper;
using System.Xml.Linq;

namespace NNFromScratch.Core.Layers
{
    public class NeuronLayer
    {
        public float[] Biases;
        public float[] NeuronValues;
        public float[] Errors;
        public float[] Weights;
        public int Size;
        public NeuronLayer PreviousLayer;
        public NeuronLayer NextLayer;
        public IActivationFunction ActivationFunction;

        public void Initialize(NeuronLayer previousLayer)
        {
            this.Biases = new float[this.Size];
            this.NeuronValues = new float[this.Size];
            this.Errors = new float[this.Size];
            this.PreviousLayer = previousLayer;

            //store the weights between the current and previous layer or null if the current layer is the input layer
            if (previousLayer != null)
                this.Weights = new float[previousLayer.Size * this.Size];

            FillRandom();
        }
        public void Summary()
        {
            //if (PreviousLayer != null)
            //    Console.WriteLine("=> Connected with " + PreviousLayer.Name + " with " + Weights.Length + " Weights");
            //Console.WriteLine("Initialize layer " + Name + " with " + Size + " Neurons");
        }
        private void FillRandom()
        {
            //Maybe use "Xavier Initialization" ref: Finn Chat DC
            for (int i = 0; i < Size; i++)
            {
                Biases[i] = MathHelper.RandomBias();
                if (Weights != null)
                    Weights[i] = MathHelper.RandomWeight();
            }
        }

        public virtual void Train()
        {
            for (int i = 0; i < this.Size; i++)
            {
                float sum = 0.0f;
                for (int j = 0; j < this.PreviousLayer.Size; j++)
                {
                    sum += this.PreviousLayer.NeuronValues[j] * this.Weights[i * this.PreviousLayer.Size + j];
                }
                this.NeuronValues[i] = MathHelper.Sigmoid(sum + this.Biases[i]);
            }
        }

        public virtual void FeedForward(float[] inputs)
        {

        }

        public virtual void Save(BinaryWriter bw)
        {
            if (Weights == null)
                return;

            bw.Write(Biases.Length);
            for (int i = 0; i < Biases.Length; i++)
            {
                bw.Write((double)Biases[i]);
            }

            bw.Write(Weights.Length);

            for (int i = 0; i < Weights.Length; i++)
            {
                bw.Write((double)Weights[i]);
            }
        }
        public virtual void Load(BinaryReader br)
        {
            int length = br.ReadInt32();
            if (length != Biases.Length)
                throw new InvalidOperationException("Weight data isn't made for this network!");
            for (int i = 0; i < length; i++)
            {
                Biases[i] = (float)br.ReadDouble();
            }
            length = br.ReadInt32();
            if (length != Weights.Length)
                throw new InvalidOperationException("Weight data isn't made for this network!");
            for (int i = 0; i < length; i++)
            {
                Weights[i] = (float)br.ReadDouble();
            }
        }
    }
}