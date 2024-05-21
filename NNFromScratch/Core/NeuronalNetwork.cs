using NNFromScratch.Helper;

namespace NNFromScratch.Core;

internal class NeuronalNetwork
{
    public readonly Neuron[] inputLayer;
    public readonly Neuron[] hiddenLayer;
    public readonly Neuron[] outputLayer;

    public NeuronalNetwork(int inputs, int hidden, int outputs)
    {
        hiddenLayer = new Neuron[hidden];
        for (int i = 0; i < hidden; i++)
        {
            hiddenLayer[i] = new Neuron(MathHelper.RandomBias());
        }

        inputLayer = new Neuron[inputs];
        for(int i = 0; i< inputs; i++)
        {
            var inputNeuron = new Neuron(MathHelper.RandomBias());
            for (int j = 0; j< hidden; j++)
            {
                NeuronLink link = new NeuronLink(MathHelper.RandomWeight(), inputNeuron, hiddenLayer[j]);
                inputNeuron.links.Add(link);
            }
            inputLayer[i] = inputNeuron;
        }

        outputLayer = new Neuron[outputs];
        for(int i = 0;i < outputs; i++)
        {
            var outputNeuron = new Neuron(MathHelper.RandomBias());
            for(int j =0; j< hidden; j++)
            {
                NeuronLink link = new NeuronLink(MathHelper.RandomWeight(), hiddenLayer[j], outputNeuron);
                outputNeuron.links.Add(link);
            }
            outputLayer[i] = outputNeuron;
        }
    }

    public double FeedForward(double[] inputs)
    {
        //var res1 = hidden1.FeedForward(inputs);
        //var res2 = hidden2.FeedForward(inputs);

        //var outpout = output1.FeedForward(new double[] { res1, res2 });
        //return outpout;
        return 0.0;
    }

    public void Draw()
    {
        int inputCount = 0;
        foreach(var inputNeuron in inputLayer)
        {
            int linkCount = 0;
            Console.WriteLine("Input" + inputCount);
            inputCount++;
            foreach(var link in inputNeuron.links)
            {
                linkCount++;
                Console.WriteLine("\tHidden" + linkCount);
            }
        }
    }
}
