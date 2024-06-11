namespace NNFromScratch.Core;

using NNFromScratch.Helper;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;

public class NNModel
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

        var trainingTime = BenchmarkExtension.Benchmark(() =>
        {
            Stopwatch epochTime = new Stopwatch();
            for (int e = 0; e < epochs; e++)
            {
                float averageStepTime = 0;
                epochTime.Restart();
                Stopwatch trainingTimeSW = new Stopwatch();
                trainingTimeSW.Start();
                {
                    for (int i = 0; i < inputs.Length; i++)
                    {
                        if (i % 1000 == 0)
                        {
                            averageStepTime += trainingTimeSW.ElapsedMilliseconds;
                            Console.WriteLine($"Epoch {e}/{epochs}; {i}/{inputs.Length}; ({trainingTimeSW.ElapsedMilliseconds}ms, {trainingTimeSW.ElapsedTicks}ticks)");
                            trainingTimeSW.Stop();
                            trainingTimeSW.Restart();
                        }
                        nn.Train(inputs[0], desired[0], 1, learningRate);
                    }
                }
                Console.WriteLine($"Epoch {e} took {epochTime.ElapsedMilliseconds}ms; avg({(int)averageStepTime / (inputs.Length / 1000)}ms/step");
            }
        });
        Console.WriteLine($"Training took: {trainingTime }");
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

    public (float percent, int count, int correct) Evaluate(float[][] x, float[][] y, bool output = true)
    {
        int correct = 0;
        for (int i = 0; i < x.Length; i++)
        {
            if(y[i][0] == MathHelper.GetMaximumIndex(nn.FeedForward(x[i])))
                correct++;
        }

        float percent = x.Length / correct;

        if(output)
            Console.WriteLine($"Evaluation: {x.Length}/{correct} ({percent})");

        return (percent, x.Length, correct);
    }
    
    public void Summary()
    {

    }
}
