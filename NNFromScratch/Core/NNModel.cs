using NNFromScratch.Core.Layers;
using NNFromScratch.Helper;
using SixLabors.ImageSharp.Formats;
using System.Diagnostics;
using System.Reflection.Emit;

namespace NNFromScratch.Core;

public class NNModel
{
    private NeuralNetwork nn;
    private bool useCuda = false;
    private bool cudaLayersInitialized;
    private BaseLayer[] layers;
    public NNModel(BaseLayer[] layers, bool useCuda = true)
    {
        this.layers = layers;

        if (layers.Length < 3)
        {
            throw new Exception("You need at least one input, hidden and output layer");
        }

        nn = new NeuralNetwork(layers);

        //use cuda only if available:
        bool hasCuda = CudaAccel.CheckCuda();
        if(useCuda && !hasCuda)
            Console.WriteLine("CUDA is not availabe");

        this.useCuda = useCuda = useCuda ? hasCuda : false;
        if (!useCuda){
            Console.WriteLine("Use CPU Compute Device");
            return;
        }

        //initialize the cuda accelerator and pass the total number of layers:
        CudaAccel.Init(layers.Length);

        //pass the references for all c# arrays to the c++ code:
        for (int i = 0; i < layers.Length; i++)
        {
            layers[i].InitializeCuda(i);
        }
    }

    public float[] Predict(float[] input, bool output = false)
    {
        float[] prediction = null;
        var time = BenchmarkExtension.Benchmark(() =>
        {
            prediction = nn.FeedForward_CPU(input);
        });
        if (output)
            Console.WriteLine("Prediction time " + time);
        return prediction;
    }

    public float[] Train(
        float[][] inputs, 
        float[][] desired, 
        int epochs, 
        float learningRate = 0.1f, 
        int loggingInterval = 100, 
        int epochInterval = 1, 
        float evaluatePercent = 10
        )
    {
        if (inputs[0].Length != nn.allLayer[0].Size)
            throw new Exception("Input size does not match input layer count");

        //let cuda check for available devices:
        if (useCuda)
            useCuda = CudaAccel.CheckCuda();

        LossCalculator lossCalc = new LossCalculator(nn);
        AccuracyCalculator accCalc = new AccuracyCalculator(nn);

        Console.WriteLine(new string('-', 50) + "\n");
        float[] accuracys = new float[epochs];
        var trainingTime = BenchmarkExtension.Benchmark(() =>
        {
            int trainDataCount = (int)(inputs.Length * (100 - evaluatePercent)) / 100;

            Stopwatch epochTime = new Stopwatch();
            Stopwatch stepTimeSW = new Stopwatch();
            for (int e = 0; e < epochs; e++)
            {
                float averageStepTime = 0;
                epochTime.Restart();
                stepTimeSW.Start();
                lossCalc.NextEpoch();
                accCalc.NextEpoch();

                for (int i = 0; i < trainDataCount; i++)
                {
                    if(useCuda)
                        CudaAccel.Train(inputs[i], desired[i], inputs.Length, learningRate);
                    else
                        nn.Train_CPU(inputs[i], desired[i], learningRate);

                    lossCalc.Calculate(desired[i]);

                    //use this, when each epoch takes longer due to more items that need to compute
                    if ((i + 1) % loggingInterval == 0)
                    {
                        stepTimeSW.Stop();

                        averageStepTime += stepTimeSW.ElapsedMilliseconds;
                        Console.WriteLine($"Epoch {e + 1}/{epochs}; {i + 1}/{trainDataCount}; ({stepTimeSW.ElapsedMilliseconds}ms, {stepTimeSW.ElapsedTicks}ticks)");
                        stepTimeSW.Restart();
                    }
                }

                //print epoch every x epochs (default: 100) => for fast training
                if ((e + 1) % epochInterval == 0)
                {
                    accCalc.Calculate(inputs, desired, trainDataCount, useCuda);

                    Console.WriteLine(new string('-', 50));
                    Console.WriteLine($"Epoch {e + 1} took {epochTime.ElapsedMilliseconds}ms; " + (averageStepTime > 0 ? $"avg({(int)averageStepTime / (inputs.Length / loggingInterval)}ms/step" : ""));
                    lossCalc.PrintLoss();
                    accCalc.PrintAccuracy();

                    //dont print the last line after training => looks weird :D
                    if (e != epochs - 1)
                        Console.WriteLine(new string('-', 50));
                }
            }
        });
        
        if (useCuda)
            CudaAccel.DoneTraining();

        Console.WriteLine(new string('=', 50) + "\n");
        Console.WriteLine($"Training took: {trainingTime}\n");

        return accuracys;
    }

    public void Save(string path)
    {
        Console.WriteLine("Saving model data to file");
        var ms = new MemoryStream();
        nn.Save(ms);
        File.WriteAllBytes(path, ms.ToArray());
        Console.WriteLine($"Saved to {path}");
    }

    public void Load(string path)
    {
        Console.WriteLine("Loading model data from file");
        var bytes = File.ReadAllBytes(path);
        var ms = new MemoryStream(bytes);
        nn.Load(ms);
        Console.WriteLine($"Loaded from {path}");

        //load the data  to the gpu
        if (this.useCuda)
        {
            //pass the references for all c# arrays to the c++ code:
            for (int i = 0; i < layers.Length; i++)
            {
                layers[i].InitializeCuda(i);
            }

            Console.WriteLine("Done offloading to GPU");
        }
    }

    //only use cuda evaluation while training, because gpu memory gets freed after training
    public (float percent, int count, int correct) Evaluate(float[][] x, float[][] y, bool predictOnCuda = true, bool output = true)
    {
        int correct = 0;

        if(predictOnCuda && !this.useCuda)
        {
            throw new Exception("Could not evaluate on GPU, because cuda was disabled in the model building");
        }

        if (predictOnCuda && this.useCuda)
        {
            for (int i = 0; i < x.Length; i++)
            {
                float[] prediction = new float[this.layers[^1].Size];
                CudaAccel.Predict(x[i], prediction);
                if (MathHelper.GetMaximumIndex(y[i]) == MathHelper.GetMaximumIndex(prediction))
                    correct++;
            }
        }
        else
        {
            for (int i = 0; i < x.Length; i++)
            {
                if (MathHelper.GetMaximumIndex(y[i]) == MathHelper.GetMaximumIndex(nn.FeedForward_CPU(x[i])))
                    correct++;
            }
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

    public void Dispose()
    {
        CudaAccel.Cleanup();
    }
}
