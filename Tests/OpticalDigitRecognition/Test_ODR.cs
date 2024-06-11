using NNFromScratch.Core;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp;
using NNFromScratch.Helper;
using NNFromScratch;

namespace Tests.TestODR;

public class Test_ODR
{
    public static void Run()
    {
        bool train = true;        

        var imageData = MNistLoader.LoadFromFile(".\\datasets\\t10k-images.idx3-ubyte", ".\\datasets\\t10k-labels.idx1-ubyte");
        int[] digits = new int[imageData.y.Length];
        int imageWidth = imageData.imageWidth;
        int imageHeight = imageData.imageHeight;

        //create the model:
        NNModel model = new NNModel(new Layer[]
        {
            new Layer(imageWidth * imageHeight, "Input"),
            new Layer(128, "Hidden1"),
            new Layer(64, "Hidden2"),
            new Layer(10, "Output"),
        });
        if (train)
        {
            //model.Load("D:\\odr1.cool");
            model.Train(imageData.x, imageData.y, epochs: 2, learningRate: 0.1f);

            Console.WriteLine(BenchmarkExtension.Benchmark(() =>
            {
                model.Evaluate(imageData.x, imageData.y);
            }));
            model.Save("D:\\odr.cool");
        }

        if (!train)
        {
            //model.Load("D:\\odr.cool");
        }
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