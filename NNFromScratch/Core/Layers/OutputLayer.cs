
namespace NNFromScratch.Core.Layers;

internal class OutputLayer : NeuronLayer
{
    public override void FeedForward()
    {


    }

    public override void Train(float[] data)
    {

        Parallel.For(0, this.Size, (i) =>
        {
            float sum = 0.0f;
            for (int j = 0; j < this.PreviousLayer.Size; j++)
            {
                int weightIndex = i * this.PreviousLayer.Size + j;
                sum += this.PreviousLayer.NeuronValues[j] * this.Weights[weightIndex];
            }
            this.NeuronValues[i] = this.ActivationFunction.Calculate(sum + this.Biases[i]);
        });
    }
}
