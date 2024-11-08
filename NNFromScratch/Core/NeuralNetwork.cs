using NNFromScratch.Core.Layers;
using NNFromScratch.Helper;

namespace NNFromScratch.Core;

internal class NeuralNetwork
{
    public readonly BaseLayer[] allLayer;

    public NeuralNetwork(BaseLayer[] layers)
    {
        this.allLayer = layers;
        for (int i = 0; i<allLayer.Length; i++)
        {
            allLayer[i].NextLayer = i + 1 < allLayer.Length ? allLayer[i + 1] : null;
            allLayer[i].PreviousLayer = i - 1 < 0 ? null : allLayer[i - 1];

            allLayer[i].Initialize(allLayer[0].Size, allLayer[allLayer.Length - 1].Size);
        }
    }

    public void Train_CPU(float[] inputs, float[] desired, float learningRate)
    {
        FeedForward_CPU(inputs);

        for(int i = allLayer.Length - 1; i >= 0; i--)
        {
            allLayer[i].Train(desired, learningRate);
        }
    }

    public float[] FeedForward_CPU(float[] data)
    {
        for (int i = 0; i < data.Length; i++)
        {
            allLayer[0].NeuronValues[i] = data[i];
        }

        foreach (var item in allLayer)
        {
            item.FeedForward();
        }

        return allLayer[allLayer.Length - 1].NeuronValues;
    }

    public void Save(Stream stream)
    {
        BinaryWriter bw = new BinaryWriter(stream);
        foreach(var l in allLayer)
        {
            l.Save(bw);
        }
        bw.Dispose();
    }

    public void Load(Stream stream)
    {
        BinaryReader br = new BinaryReader(stream);
        foreach (var l in allLayer)
        {
            l.Load(br);
        }
        br.Dispose();
    }

    public void Summary()
    {
        Console.WriteLine(new string('-', 50));
        foreach(var layer in allLayer)
        {
            layer.Summary();
        }
        Console.WriteLine(new string('=', 50));
    }
}
