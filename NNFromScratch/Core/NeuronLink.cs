
namespace NNFromScratch.Core;

internal class NeuronLink
{
    public double Weight;
    public Neuron LeftNeuron;
    public Neuron RightNeuron;

    public NeuronLink(double weight, Neuron leftNeuron, Neuron rightNeuron)
    {
        Weight = weight;
        LeftNeuron = leftNeuron;
        RightNeuron = rightNeuron;
    }
}
