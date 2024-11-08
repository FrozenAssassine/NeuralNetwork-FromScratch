using NNFromScratch.Core.Layers;
using NNFromScratch.Core;
using NNFromScratch.Models;
using NNFromScratch.Helper;

namespace Tests.CNN;

internal class Test_CNN
{
    //dataset from: https://www.kaggle.com/datasets/nitishabharathi/scene-classification
    const string datasetPath = "D:\\testnn\\scene_classification\\train";
    const string csvPath = "D:\\testnn\\scene_classification\\train.csv";
    const int ImageWidth = 150;
    const int ImageHeight = 150;
    const int PixelDepth = 3; //rgb
    const int ImageCount = 500; //total of 17033 images
    const int OutputTypes = 6; //Buildings, Forests, Mountains, Glacier, Street, Sea

    public static void Run()
    {
        float[][] images = ImageHelper.GetImages(datasetPath, ImageCount, ImageWidth, ImageHeight);
        int[] csvRow1 = CSVHelper.GetCSVRow_Int(csvPath, 1, ImageCount, ".jpg"); //exampleData: 249.jpg,5

        //make the data
        float[][] desired = new float[csvRow1.Length][];
        for (int i = 0; i < csvRow1.Length; i++)
        {
            float[] res = new float[OutputTypes];
            for (int j = 0; j < res.Length; j++)
            {
                res[j] = j == csvRow1[i] ? 1 : 0;
            }
            desired[i] = res;
        }


        var convLayer = new ConvolutionalLayer(ImageWidth, ImageHeight, 1, 3, 3, 5);
        var poolLayer = new PoolingLayer(ImageWidth, ImageHeight, convLayer.featureMapX, convLayer.featureMapY, 3, 5);


        //create the neural network:
        var network = NetworkBuilder.Create()
            .Stack(new InputLayer(ImageWidth * ImageHeight * PixelDepth))
            .Stack(convLayer)
            .Stack(poolLayer)
            .Stack(new DenseLayer(poolLayer.Size, ActivationType.Relu))
            .Stack(new OutputLayer(6, ActivationType.Softmax))
            .Build(false);

        network.Summary();

        //network.Load("D:\\imageclassification2.cool");

        network.Train(images, desired, 10, 0.01f);

        Console.WriteLine("Press enter to save");
        Console.ReadLine();
        network.Save("D:\\imageclassification2.cool");

        Console.WriteLine("Press enter to evaluate");
        Console.ReadLine();
        Random random = new Random();

        var randomIndices = Enumerable.Range(0, 500)
                                      .Select(_ => random.Next(0, images.Length))
                                      .Distinct()
                                      .ToList();

        var randImages = randomIndices.Select(index => images[index]);
        var randDesired = randomIndices.Select(index => desired[index]);

        network.Evaluate(randImages.ToArray(), randDesired.ToArray(), false);
    }
}
