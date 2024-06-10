using Test1.Data1;

public class Run1
{
    public static void main(String[] args)
    {
        string imagePath = "D:\\testnn\\t10k-images.idx3-ubyte";
        string labelPath = "D:\\testnn\\t10k-labels.idx1-ubyte";

        DigitData[] data = MNistLoader.LoadFromFile(imagePath, labelPath);

        int[] digits = new int[data.Length];

        DigitRecognition recognizer = new DigitRecognition(data[0].Width, data[0].Height);

        //recognizer.Train(data, 1, 0.1f);

        recognizer.Load("D:\\testnn\\odr.cool");

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

        //recognizer.Save("D:\\testnn\\odr.cool");
        
        Console.WriteLine("Correct: " + correct + "/" + data.Length);
    }
}

//Correct: 8388/10000