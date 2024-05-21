namespace NNFromScratch.Core;
internal class Neuron
{
    public double bias;
    public List<NeuronLink> links = new List<NeuronLink>();

    public Neuron(NeuronLink link, double bias)
    {
        this.bias = bias;
        this.links.Add(link);
    }

    public Neuron(double bias)
    {
        this.bias = bias;
    }

    public double FeedForward(double[] inputs)
    {
        return 0; // return MathHelper.Sigmoid(MathHelper.DotProduct(inputs, weights) + bias);
    }
}
