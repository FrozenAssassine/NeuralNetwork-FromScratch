namespace NNFromScratch.Core;

using NNFromScratch.Helper;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;

internal class NNModel
{
    private NeuralNetwork nn;

    public NNModel(Layer[] layers)
    {
        if(layers.Length < 3)
        {
            throw new Exception("You need at least one input, hidden and output layer");
        }

        var hidden = layers.Length == 1 ? layers.Skip(1) : layers.Skip(1).Take(layers.Length - 2);
        nn = new NeuralNetwork(layers[0], hidden.ToArray(), layers[layers.Length - 1]);
    }

    public float[] Predict(float[] input, bool output = false)
    {
        float[] prediction = null;
        var time = BenchmarkExtension.Benchmark(() =>
        {
            prediction = nn.FeedForward(input);
        });

        if (output)
            Console.WriteLine("Prediction time " + time);

        return prediction;
    }

    public void Train(float[][] inputs, float[][] desired, int epochs, float learningRate = 0.01f)
    {
        if (inputs[0].Length != nn.inputLayer.Size)
            throw new Exception("Input size does not match input layer count");

        if (desired[0].Length != nn.outputLayer.Size)
            throw new Exception("Desired size does not match output layer count");

        float printCount = inputs[0].Length / 1000 < 1 ? 1 : 1000; 

        for(int e = 0; e<epochs; e++)
        {
            Stopwatch trainingTimeSW = new Stopwatch();
            trainingTimeSW.Start();
            {
                for (int i = 0; i < inputs.Length; i++)
                {
                    if (i % printCount == 0)
                    {
                        Console.WriteLine($"Epoch {e}/{epochs}; {i}/{inputs.Length}; ({trainingTimeSW.ElapsedMilliseconds}ms, {trainingTimeSW.ElapsedTicks}ticks)");
                        trainingTimeSW.Stop();
                        trainingTimeSW.Restart();
                    }
                    nn.Train(inputs[i], desired[i], 1, learningRate);
                }
            }
        }
    }

    public void Save(string path)
    {
        var ms = new MemoryStream();
        nn.Save(ms);
        File.WriteAllBytes(path, ms.ToArray());
    }

    public void Load(string path)
    {
        var bytes = File.ReadAllBytes(path);
        var ms = new MemoryStream(bytes);
        nn.Load(ms);
    }

    public void Summary()
    {

    }
}
