using NNFromScratch;
using NNFromScratch.Core;
using Test1.Data1;

public class Program
{
    public static void Main()
    {
        NNModel nnmodel = new NNModel(new Layer[]
        {
            new Layer(2),
            new Layer(4),
            new Layer(1),
        });

        float[][] inputs = new float[][] { new float[] { 0, 0 }, new float[] { 0, 1 }, new float[] { 1, 0 }, new float[] { 1, 1 } };
        float[][] desired = new float[][] { new float[] { 0 }, new float[] { 1 }, new float[] { 1 }, new float[] { 0 } };
        nnmodel.Train(inputs, desired, 16000, 0.01f);

        var predict = nnmodel.Predict(new float[] { 0, 0 });
        foreach(var pred in predict)
        {
            Console.WriteLine(pred);
        }

        return;
        bool train = false;
        string imagePath = ".\\datasets\\t10k-images.idx3-ubyte";
        string labelPath = ".\\datasets\\t10k-labels.idx1-ubyte";
        DigitData[] data = MNistLoader.LoadFromFile(imagePath, labelPath);
        int[] digits = new int[data.Length];
        DigitRecognition recognizer = new DigitRecognition(data[0].Width, data[0].Height);

        if (train)
        {
            string time = BenchmarkExtension.Benchmark(() =>
            {
                recognizer.Train(data, 20, 0.1f);
            });
            Console.WriteLine("Training took " + time);

            for (int i = 0; i < digits.Length; i++)
            {
                digits[i] = data[i].Digit;
                data[i].Digit = -1;
            }

            recognizer.Classify(data);

            int correct = 0;
            for (int i = 0; i < data.Length; i++)
            {
                if (data[i].Digit == digits[i])
                    correct++;
            }
            Console.WriteLine("Test: " + correct + "/" + data.Length + " correct classifications");
            recognizer.Save("D:\\odr.cool");
        }

        if (!train)
        {
            recognizer.Load("D:\\odr.cool");
            recognizer.Classify(".\\test\\2.png", true);
        }
    }
}