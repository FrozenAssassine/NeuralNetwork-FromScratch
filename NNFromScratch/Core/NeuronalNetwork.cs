using NNFromScratch.Helper;
using System.Diagnostics;
using System.Runtime.CompilerServices;

namespace NNFromScratch.Core;

internal class NeuronalNetwork
{
    public readonly Neuron[] inputLayer;
    public readonly Neuron[] hiddenLayer;
    public readonly Neuron[] outputLayer;

    public NeuronalNetwork(int inputs, int hidden, int outputs)
    {
        //initialize the allWeights array:
        WeightHelper.Init(inputs * hidden + hidden * outputs);
        Console.WriteLine("Number of weights: " + WeightHelper.AllWeights.Length);

        int weightIndexCounter = 0;

        inputLayer = new Neuron[inputs];
        for(int i = 0; i< inputs; i++)
        {
            inputLayer[i] = new Neuron(weightIndexCounter++, MathHelper.RandomBias());
        }

        hiddenLayer = new Neuron[hidden];
        for (int i = 0; i < hidden; i++)
        {
            hiddenLayer[i] = new Neuron(weightIndexCounter++, MathHelper.RandomBias());
        }

        outputLayer = new Neuron[outputs];
        for (int i = 0;i < outputs; i++)
        {
            outputLayer[i] = new Neuron(weightIndexCounter++, MathHelper.RandomBias());
        }
    }

    public double FeedForward(double[] data)
    {
        if (data.Length != inputLayer.Length)
            throw new Exception("Input size is not the same as the number of layers");

        //fill the input neurons with its corresponding data
        for (int i = 0; i < data.Length; i++)
        {
            inputLayer[i].value = data[i];
        }

        for (int i = 0; i < data.Length; i++)
        {
            hiddenLayer[i].value = hiddenLayer[i].FeedForward(inputLayer);
        }
        return 0.0;
    }
}
