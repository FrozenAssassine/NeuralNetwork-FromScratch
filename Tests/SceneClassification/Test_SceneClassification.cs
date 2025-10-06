using NNFromScratch.Core;
using NNFromScratch.Core.Layers;
using NNFromScratch.Helper;

namespace Tests.SceneClassification
{
    internal class Test_SceneClassification
    {
        //dataset from: https://www.kaggle.com/datasets/nitishabharathi/scene-classification
        const string datasetPath = "D:\\testnn\\scene_classification\\train";
        const string csvPath = "D:\\testnn\\scene_classification\\train.csv";
        const int ImageWidth = 150;
        const int ImageHeight = 150;
        const int PixelDepth = 3; //rgb
        const int ImageCount = 1000; //total of 17033 images
        const int OutputTypes = 6; //Buildings, Forests, Mountains, Glacier, Street, Sea
        public static void Run()
        {
            float[][] images = ImageHelper.GetImages(datasetPath, ImageCount, ImageWidth, ImageHeight);
            int[] csvRow1 = CSVHelper.GetCSVRow_Int(csvPath, 1, ImageCount, ".jpg"); //exampleData: 249.jpg,5

            //make the data
            float[][] desired = new float[csvRow1.Length][];
            for(int i = 0; i< csvRow1.Length; i++)
            {
                float[] res = new float[OutputTypes];
                for (int j = 0; j < res.Length; j++)
                {
                    res[j] = j == csvRow1[i] ? 1 : 0;
                }
                desired[i] = res;
            }

            var network = NetworkBuilder.Create()
                .Stack(new InputLayer(ImageWidth * ImageHeight * PixelDepth))
                .Stack(new DenseLayer(1024, ActivationType.ReLU))
                .Stack(new DenseLayer(512, ActivationType.ReLU))
                .Stack(new DenseLayer(256, ActivationType.ReLU))
                .Stack(new OutputLayer(OutputTypes, ActivationType.Softmax))
                .Build();

            //network.Load("D:\\imageclassification.cool");

            network.Train(images, desired, 2,0.1f);
            Console.WriteLine("Press enter to Save");
            Console.ReadLine();

            //network.Save("D:\\imageclassification.cool");

            Random random = new Random();

            var randomIndices = Enumerable.Range(0, 1000)
                                          .Select(_ => random.Next(0, images.Length))
                                          .Distinct()
                                          .ToList();

            var randImages = randomIndices.Select(index => images[index]);
            var randDesired = randomIndices.Select(index => desired[index]);

            network.Evaluate(randImages.ToArray(), randDesired.ToArray(), false);
        }
    }
}
