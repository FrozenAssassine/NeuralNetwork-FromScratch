using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNFromScratch.Core.Layers
{
    public class RNNLayer : BaseLayer
    {
        private float[] hiddenState;
        private float[] hiddenWeights;
        private int inputSize;
        private int hiddenSize;

        public RNNLayer(int inputSize, int hiddenSize)
        {
            this.inputSize = inputSize;
            this.hiddenSize = hiddenSize;
            this.Size = hiddenSize;
            this.hiddenState = new float[hiddenSize];
            this.Weights = new float[inputSize * hiddenSize]; // Input-to-hidden weights
            this.hiddenWeights = new float[hiddenSize * hiddenSize]; // Hidden-to-hidden weights
            this.Biases = new float[hiddenSize];
            Initialize(); // Initialize weights and biases
        }

        public override void FeedForward()
        {
            // Zero the current neuron values
            Array.Clear(NeuronValues, 0, hiddenSize);

            // Compute the input-to-hidden activation
            for (int i = 0; i < hiddenSize; i++)
            {
                float activation = Biases[i];
                for (int j = 0; j < inputSize; j++)
                {
                    activation += PreviousLayer.NeuronValues[j] * Weights[i * inputSize + j];
                }

                // Add the hidden-to-hidden recurrent connection
                for (int h = 0; h < hiddenSize; h++)
                {
                    activation += hiddenState[h] * hiddenWeights[i * hiddenSize + h];
                }

                // Apply the activation function (e.g., tanh)
                NeuronValues[i] = (float)Math.Tanh(activation);
            }

            // Update hidden state
            Array.Copy(NeuronValues, hiddenState, hiddenSize);
        }

        public override void Train(float[] desiredValues, float learningRate)
        {
            // Error calculation: delta = desiredValues - NeuronValues
            for (int i = 0; i < hiddenSize; i++)
            {
                Errors[i] = desiredValues[i] - NeuronValues[i];
            }

            // Update weights and biases (simple gradient descent)
            for (int i = 0; i < hiddenSize; i++)
            {
                // Update biases
                Biases[i] += learningRate * Errors[i];

                // Update input-to-hidden weights
                for (int j = 0; j < inputSize; j++)
                {
                    Weights[i * inputSize + j] += learningRate * Errors[i] * PreviousLayer.NeuronValues[j];
                }

                // Update hidden-to-hidden weights
                for (int h = 0; h < hiddenSize; h++)
                {
                    hiddenWeights[i * hiddenSize + h] += learningRate * Errors[i] * hiddenState[h];
                }
            }
        }

        public override void Initialize()
        {
            Random rand = new Random();
            // Initialize weights and biases randomly
            for (int i = 0; i < Weights.Length; i++)
            {
                Weights[i] = (float)(rand.NextDouble() - 0.5); // Random values between -0.5 and 0.5
            }

            for (int i = 0; i < hiddenWeights.Length; i++)
            {
                hiddenWeights[i] = (float)(rand.NextDouble() - 0.5);
            }

            for (int i = 0; i < Biases.Length; i++)
            {
                Biases[i] = 0; // Initialize biases to 0
            }
        }

        public override void Summary()
        {
            Console.WriteLine($"RNN Layer: Input Size = {inputSize}, Hidden Size = {hiddenSize}");
        }

        public override void Save(BinaryWriter bw)
        {
            // Save weights and biases to a file
            foreach (var weight in Weights)
            {
                bw.Write(weight);
            }
            foreach (var hWeight in hiddenWeights)
            {
                bw.Write(hWeight);
            }
            foreach (var bias in Biases)
            {
                bw.Write(bias);
            }
        }

        public override void Load(BinaryReader br)
        {
            // Load weights and biases from a file
            for (int i = 0; i < Weights.Length; i++)
            {
                Weights[i] = br.ReadSingle();
            }
            for (int i = 0; i < hiddenWeights.Length; i++)
            {
                hiddenWeights[i] = br.ReadSingle();
            }
            for (int i = 0; i < Biases.Length; i++)
            {
                Biases[i] = br.ReadSingle();
            }
        }

        public override void InitializeCuda(int index)
        {
            // CUDA initialization can be implemented here
            throw new NotImplementedException();
        }
    }
}