using NNFromScratch.Helper;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNFromScratch.Core
{
    internal class OutputLayer : Layer
    {
        public OutputLayer(int size, string name) : base(size, name)
        {
        }

        public override float[] FeedForward()
        {
            for (int i = 0; i < this.Size; i++)
            {
                for (int j = 0; j < this.PreviousLayer.Size; j++)
                {
                    this.NeuronValues[i] = MathHelper.Sigmoid(this.Weight[i * this.PreviousLayer.NeuronValues.Length + j] + this.Biases[i]);
                }

            }
        }

        public override void Train(float learningRate)
        {
            for (int i = 0; i < Errors.Length; i++)
            {
                double followingWeightedErrors = 0;
                for (int j = 0; j < followingLayer.NeuronValues.Length; j++)
                {
                    followingWeightedErrors += followingLayer.Errors[j] * followingLayer.Weight[j, i];
                }
                Errors[i] = MathHelper.SigmoidDerivative(NeuronValues[i]);

                for (int j = 0; j < PreviousLayer.NeuronValues.Length j++)
                {
                    Weight[i, j] += learningRate * Errors[i] * PreviousLayer.NeuronValues[j];
                }
                Biases[i] += learningRate * Errors[i];
            }
        }
    }
}
