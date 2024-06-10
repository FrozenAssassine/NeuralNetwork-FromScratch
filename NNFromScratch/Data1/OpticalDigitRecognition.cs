using NNFromScratch;
using NNFromScratch.Core;
using System;
using System.Collections.Generic;
using System.Diagnostics;

namespace Test1.Data1
{
    public class DigitRecognition
    {
        NeuralNetwork nn;

        public DigitRecognition(int imageWidth, int imageHeight)
        {
            //NeuralNetwork.ChunkSize = SIMDAccelerator.SIMDLength;

            Layer input = new Layer(imageWidth * imageHeight, "Input");
            Layer hidden1 = new Layer(128, "Hidden1");
            Layer hidden2 = new Layer(64, "Hidden2");
            Layer output = new Layer(10, "Output");

            nn = new NeuralNetwork(input, new Layer[] { hidden1, hidden2 }, output);
            //network = new NeuralNetwork(imageWidth * imageHeight, 2, 10, 20);
            //network.Accelerator = new SIMDAccelerator();
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
                            Console.WriteLine(j + "/" + epochs + ", " + i + "/" + data.Length + ": " + $"{sw.ElapsedMilliseconds}ms ({sw.ElapsedTicks}ticks)");
                            sw.Stop();
                            sw.Restart();
                        }
                        nn.Train2(Prepare(data[i].Data), GetDigitArrayFromDigit(data[i].Digit), 2, learningRate);
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
    }
}