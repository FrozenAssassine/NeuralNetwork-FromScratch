using NNFromScratch.Helper;

namespace NNFromScratch.Core;
internal class Neuron
{
    public double bias;
    public double value;
    public int weightIndex;

    public Neuron(int weightIndex, double bias)
    {
        this.bias = bias;
        this.weightIndex = weightIndex;
    }

    public Neuron(double bias)
    {
        this.bias = bias;
    }

    public double FeedForward(Neuron[] inputs)
    {
        return MathHelper.Sigmoid(MathHelper.DotProduct(inputs) + bias);
    }
}
