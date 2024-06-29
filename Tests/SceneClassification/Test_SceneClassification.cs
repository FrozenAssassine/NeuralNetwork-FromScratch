using NNFromScratch.Core;
using NNFromScratch.Core.ActivationFunctions;
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
        const int ImageCount = 1000; //total of 24335 images
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

            var activation = new SigmoidActivation();
            var network = NetworkBuilder.Create()
                .Stack(new InputLayer(ImageWidth * ImageHeight * PixelDepth, activation))
                .Stack(new NeuronLayer(1024, activation))
                .Stack(new NeuronLayer(512, activation))
                .Stack(new OutputLayer(OutputTypes, activation))
                .Build();

            //network.Load("D:\\imageclassification_cpu.cool");

            network.Train(images, desired, 3,0.1f, true);
            Console.WriteLine("Press enter to Save");
            Console.ReadLine();

            //network.Save("D:\\imageclassification.cool");

            network.Evaluate(images.Take(1000).ToArray(), desired.Take(1000).ToArray(), false);
        }
    }
}
