using NNFromScratch.Helper;
using System.Diagnostics;

namespace NNFromScratch.Core;

public class NNModel
{
    private NeuralNetwork nn;

    public NNModel(Layer[] layers)
    {
        if (layers.Length < 3)
        {
            throw new Exception("You need at least one input, hidden and output layer");
        }

        var hidden = layers.Length == 1 ? layers.Skip(1) : layers.Skip(1).Take(layers.Length - 2);
        nn = new NeuralNetwork(layers[0], hidden.ToArray(), layers[layers.Length - 1]);

        //initialize the cuda accelerator and pass the total number of layers:
        CudaAccel.Init(layers.Length);

        //pass the references for all c# arrays to the c++ code:
        int layerIndex = 0;
        foreach (var layer in layers)
        {
            int prevSize = layerIndex > 0 ? layers[layerIndex - 1].Size : 0;
            CudaAccel.InitLayer(layerIndex++, prevSize, layer.Size, layer.Biases, layer.Weights, layer.NeuronValues, layer.Errors);
        }
    }

    public float[] Predict(float[] input, bool output = false)
    {
        float[] prediction = null;
        var time = BenchmarkExtension.Benchmark(() =>
        {
            prediction = nn.FeedForward(input, false);
        });

        if (output)
            Console.WriteLine("Prediction time " + time);

        return prediction;
    }

    public float[] Train(float[][] inputs, float[][] desired, int epochs, float learningRate = 0.1f, bool useCuda = true, bool evaluate = false, int evaluatePercent = 10)
    {
        if (inputs[0].Length != nn.inputLayer.Size)
            throw new Exception("Input size does not match input layer count");

        //let cuda check for available devices:
        if (useCuda)
        {
            useCuda = CudaAccel.CheckCuda();
        }

        int loggingInterval = 1000;

        Console.WriteLine(new string('-', 50) + "\n");
        float[] accuracys = new float[epochs];

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
                        nn.Train(inputs[i], desired[i], learningRate, useCuda);

                        if ((i+1) % loggingInterval == 0)
                        {
                            trainingTimeSW.Stop();

                            averageStepTime += trainingTimeSW.ElapsedMilliseconds;
                            Console.WriteLine($"Epoch {e + 1}/{epochs}; {i + 1}/{inputs.Length}; ({trainingTimeSW.ElapsedMilliseconds}ms, {trainingTimeSW.ElapsedTicks}ticks)");
                            
                            trainingTimeSW.Restart();
                        }
                    }
                }

                Console.WriteLine(new string('-', 50));
                Console.WriteLine($"Epoch {e + 1} took {epochTime.ElapsedMilliseconds}ms; " + (averageStepTime > 0 ? $"avg({(int)averageStepTime / (inputs.Length / loggingInterval)}ms/step" : ""));

                //evaluate after every epoch
                if (evaluate)
                {
                    int percent = inputs.Length / 100 * evaluatePercent;
                    accuracys[e] = percent;
                    Evaluate(inputs.Take(percent).ToArray(), desired.Take(percent).ToArray(), false);
                }

                if (e != epochs - 1)
                    Console.WriteLine(new string('-', 50));
            }
        });
        //important to free memory from gpu
        if (useCuda)
            CudaAccel.DoneTraining();

        Console.WriteLine(new string('=', 50) + "\n");
        Console.WriteLine($"Training took: {trainingTime}\n");

        return accuracys;
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

    //only use cuda evaluation while training, because gpu memory gets freed after training
    public (float percent, int count, int correct) Evaluate(float[][] x, float[][] y, bool useCuda, bool output = true)
    {
        int correct = 0;
        for (int i = 0; i < x.Length; i++)
        {
            if (MathHelper.GetMaximumIndex(y[i]) == MathHelper.GetMaximumIndex(nn.FeedForward(x[i], useCuda)))
                correct++;
        }

        float accuracy = (float)correct / x.Length;
        if(output)
            Console.WriteLine($"Evaluation: {x.Length}/{correct} ({accuracy.ToString().Replace(",", ".")}) ({(int)(accuracy * 100.0f)}%)");

        return (accuracy, x.Length, correct);
    }
    
    public void Summary()
    {
        nn.Summary();
    }
}
