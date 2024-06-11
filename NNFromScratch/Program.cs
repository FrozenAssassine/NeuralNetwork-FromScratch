using NNFromScratch;
using Test1.Data1;

public class Program
{
    public static void Main()
    {
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