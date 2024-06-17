namespace NNFromScratch.Core.Layers
{
    internal class InputLayer : NeuronLayer
    {
        public InputLayer()
        {

        }

        public void SetInputs(float[] inputs)
        {
            //TODO put elsewhere for
            //if (inputs.Length != this.Size)
            //    throw new Exception("Input size is not the same as the number of layers");

            for (int i = 0; i < inputs.Length; i++)
            {
                this.NeuronValues[i] = inputs[i];
            }
        }

        public override void FeedForward()
        {

        }

        //Compute neuron values for output layer
        public override void Train()
        {

        }
    }
}
