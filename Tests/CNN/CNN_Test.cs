using NNFromScratch.Core.Layers;
using NNFromScratch.Core;
using NNFromScratch;
using Tests.TestODR;

namespace Tests.CNN;

internal class CNN_Test
{
    public static void Run()
    {
        var trainData = MNistLoader.LoadFromFile(".\\datasets\\train-images.idx3-ubyte", ".\\datasets\\train-labels.idx1-ubyte");
        //var trainData = MNistLoader.LoadFromFile(".\\datasets\\t10k-images.idx3-ubyte", ".\\datasets\\t10k-labels.idx1-ubyte");
        int[] digits = new int[trainData.y.Length];
        int imageWidth = trainData.imageWidth;
        int imageHeight = trainData.imageHeight;

        //create the neural network:
        var network = NetworkBuilder.Create()
            .Stack(new InputLayer(28 * 28))
            .Stack(new ConvolutionalLayer(32, (28, 28), 3, 0, ActivationType.Relu))
            .Stack(new MaxPoolingLayer(2, 2, (28, 28)))
            .Stack(new ConvolutionalLayer(64, (28, 28), 3, 0, ActivationType.Relu))
            .Stack(new MaxPoolingLayer(2, 2, (28, 28)))
            .Stack(new OutputLayer(10, ActivationType.Softmax))
            .Build(false);

        network.Summary();
        //network.Load("D:\\odr.cool");
        network.Train(trainData.x.Take(10000).ToArray(), trainData.y.Take(10000).ToArray(), epochs: 10, learningRate: 0.01f, 1000, 1, 5);

        Console.WriteLine(BenchmarkExtension.Benchmark(() =>
        {
            network.Evaluate(trainData.x, trainData.y, false);
        }));

        Console.WriteLine("Press Enter to Save");
        Console.ReadLine();

        network.Save("D:\\odr_good.cool");

    }
}