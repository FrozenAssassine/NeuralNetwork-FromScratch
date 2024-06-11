using NNFromScratch.Core;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp;
using System.Diagnostics;
using NNFromScratch.Helper;
using NNFromScratch;

namespace Test1.Data1
{
    public class DigitRecognition
    {
        NeuralNetwork nn;

        public DigitRecognition(int imageWidth, int imageHeight)
        {
            Layer input = new Layer(imageWidth * imageHeight, "Input");
            Layer hidden1 = new Layer(128, "Hidden1");
            Layer hidden2 = new Layer(64, "Hidden2");
            Layer output = new Layer(10, "Output");

            nn = new NeuralNetwork(input, new Layer[] { hidden1, hidden2 }, output);
        }

        public void Train(DigitData[] data, int epochs, float learningRate = 0.01f)
        {
            for (int j = 0; j < epochs; j++)
            {
                Stopwatch sw = new Stopwatch();
                sw.Start();
                {
                    for (int i = 0; i < data.Length; i++)
                    {
                        if (i % 1000 == 0)
                        {
                            Console.WriteLine($"Epoch {j}/{epochs}; {i}/{data.Length}; ({sw.ElapsedMilliseconds}ms, {sw.ElapsedTicks}ticks)");
                            sw.Stop();
                            sw.Restart();
                        }
                        nn.Train(Prepare(data[i].Data), GetDigitArrayFromDigit(data[i].Digit), 1, learningRate);
                    }
                }
            }
        }

        public void Classify(DigitData[] data)
        {
            for (int i = 0; i < data.Length; i++)
            {
                data[i].Digit = GetDigitFromDigitArray(nn.FeedForward(Prepare(data[i].Data)));
            }
        }

        public void Classify(string path, bool showPredictions = false)
        {
            int maxIndex = 0;
            float[] prediction = null;
            var time = BenchmarkExtension.Benchmark(() =>
            {
                var image = GetImagePixel(path);
                prediction = nn.FeedForward(image);
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

        private float[] Prepare(byte[] bytes)
        {
            float[] res = new float[bytes.Length];
            for (int i = 0; i < bytes.Length; i++)
            {
                res[i] = bytes[i] / 255.0f;
            }
            return res;
        }

        private float[] GetDigitArrayFromDigit(int digit)
        {
            float[] res = new float[10];
            for (int i = 0; i < res.Length; i++)
            {
                if (i == digit)
                    res[i] = 1;
                else
                    res[i] = 0;
            }
            return res;
        }

        private int GetDigitFromDigitArray(float[] d)
        {
            int maxIndex = 0;
            float max = d[0];
            for (int i = 1; i < d.Length; i++)
            {
                if (max < d[i])
                {
                    maxIndex = i;
                    max = d[i];
                }
            }
            return maxIndex;
        }

        public void Save(string path)
        {
            var ms = new MemoryStream();
            nn.Save(ms);
            File.WriteAllBytes(path, ms.ToArray());
        }

        public void Load(string path)
        {
            var bytes = File.ReadAllBytes(path);
            var ms = new MemoryStream(bytes);
            nn.Load(ms);
        }

        //returns 0 for all colors and 1 for all black colors with alpha of exactly 255
        public float[] GetImagePixel(string path)
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
    }
}