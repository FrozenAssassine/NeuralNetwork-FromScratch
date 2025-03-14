﻿using NNFromScratch.Core;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp;
using NNFromScratch.Helper;
using NNFromScratch;
using NNFromScratch.Core.Layers;

namespace Tests.TestODR;

public class Test_ODR
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
            .Stack(new InputLayer(imageWidth * imageHeight))
            .Stack(new DenseLayer(512, ActivationType.Sigmoid))
            .Stack(new DenseLayer(256, ActivationType.Sigmoid))
            .Stack(new OutputLayer(10, ActivationType.Softmax))
            .Build(true);

        network.Summary();
        //network.Load("D:\\odr.cool");
        network.Train(trainData.x, trainData.y, epochs: 10, learningRate: 0.01f, 1000, 1, 5);

        Console.WriteLine(BenchmarkExtension.Benchmark(() =>
        {
            network.Evaluate(trainData.x, trainData.y, false);
        }));

        Console.WriteLine("Press Enter to Save");
        Console.ReadLine();

        network.Save("D:\\odr_good.cool");
    }

    //returns 0 for all colors and 1 for all black colors with alpha of exactly 255
    public static float[] GetImagePixel(string path)
    {
        float[] pixels = new float[28 * 28];
        int index = 0;
        using Image<Rgba32> image = Image.Load<Rgba32>(path);
        image.ProcessPixelRows(accessor =>
        {
            Rgba32 transparent = Color.Transparent;

            for (int y = 0; y < accessor.Height; y++)
            {
                Span<Rgba32> pixelRow = accessor.GetRowSpan(y);

                for (int x = 0; x < pixelRow.Length; x++)
                {
                    ref Rgba32 pixel = ref pixelRow[x];
                    pixels[index++] = (pixel.R == 0 && pixel.G == 0 && pixel.B == 0 && pixel.A == 255) ? 1 : 0;
                }
            }
        });
        return pixels;
    }
    private static void Classify(NNModel nn, string path, bool showPredictions = false)
    {
        int maxIndex = 0;
        float[] prediction = null;
        var time = BenchmarkExtension.Benchmark(() =>
        {
            var image = GetImagePixel(path);
            prediction = nn.Predict(image);
            maxIndex = MathHelper.GetMaximumIndex(prediction);
        });

        if (prediction == null)
            return;

        if (showPredictions)
        {
            Console.WriteLine("Predictions:");
            for (int i = 0; i < prediction.Length; i++)
            {
                var pred = prediction[i];
                Console.WriteLine($"\t {i}: {MathF.Round(pred, 4)}");
            }
        }

        Console.WriteLine($"Provided image is {maxIndex} with probability {prediction[maxIndex]}; Took {time}");
    }
}