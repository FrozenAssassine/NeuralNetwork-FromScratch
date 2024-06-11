using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNFromScratch.Core
{
    internal class InputLayer : Layer
    {
        public void SetInputs(float[] inputs)
        {
            if (inputs.Length != this.Size)
                throw new Exception("Input size is not the same as the number of layers");

            //fill the input neurons with its corresponding data
            for (int i = 0; i < inputs.Length; i++)
            {
                this.NeuronValues[i] = inputs[i];
            }

            followingLayer.FeedForward();
        }

        public InputLayer(int size, string name) : base(size, name)
        {

        }

        public override void Train(float learningRate)
        {

        }

        public override float[] FeedForward()
        {
            return null;
        }
    }
}
