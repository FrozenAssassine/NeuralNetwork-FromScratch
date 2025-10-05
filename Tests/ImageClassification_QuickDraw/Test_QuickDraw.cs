using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Newtonsoft.Json;
using NNFromScratch.Core.Layers;
using NNFromScratch.Core;

namespace Tests.ImageClassification_QuickDraw
{
    public class Drawing
    {
        public Drawing(DrawingEntry[] drawings)
        {
            this.Drawings = drawings;
        }

        public readonly DrawingEntry[] Drawings;
    }

    public class DrawingEntry
    {
        [JsonProperty("key_id")]
        public ulong KeyId { get; set; }

        [JsonProperty("word")]
        public string Word { get; set; }

        [JsonProperty("recognized")]
        public bool Recognized { get; set; }

        [JsonProperty("timestamp")]
        public string Timestamp { get; set; }

        [JsonProperty("countrycode")]
        public string CountryCode { get; set; }

        [JsonProperty("drawing")]
        public List<List<List<double>>> Drawing { get; set; }

        public List<List<Point>> GetPointArray()
        {
            var points = new List<List<Point>>();
            foreach (var stroke in Drawing)
            {
                List<Point> points2 = new();
                points.Add(points2);

                if (stroke.Count >= 3)
                {
                    for (int i = 0; i < stroke[0].Count; i++)
                    {
                        points2.Add(new Point
                        {
                            X = stroke[0][i],
                            Y = stroke[1][i],
                            T = (int)stroke[2][i]
                        });
                    }
                }
            }
            return points;
        }

        public void ResizeDrawing(double newWidth, double newHeight)
        {
            if (Drawing == null || Drawing.Count == 0) return;

            double minX = Drawing.SelectMany(stroke => stroke[0]).Min();
            double maxX = Drawing.SelectMany(stroke => stroke[0]).Max();
            double minY = Drawing.SelectMany(stroke => stroke[1]).Min();
            double maxY = Drawing.SelectMany(stroke => stroke[1]).Max();

            double scaleX = newWidth / (maxX - minX);
            double scaleY = newHeight / (maxY - minY);

            foreach (var stroke in Drawing)
            {
                for (int i = 0; i < stroke[0].Count; i++)
                {
                    stroke[0][i] = (stroke[0][i] - minX) * scaleX;
                    stroke[1][i] = (stroke[1][i] - minY) * scaleY;
                }
            }
        }
    }

    public class Point
    {
        public double X { get; set; }
        public double Y { get; set; }
        public int T { get; set; }
    }

    public class NdjsonParser
    {
        public static IEnumerable<Drawing> IterateFiles(string folderPath, int maxIterations)
        {
            foreach (var filePath in Directory.GetFiles(folderPath))
            {
                yield return new Drawing(NdjsonParser.ParseNdjson(filePath, maxIterations).ToArray());
            }
        }

        public static IEnumerable<DrawingEntry> ParseNdjson(string filePath, int maxIterations)
        {
            int count = 0;
            using (var reader = new StreamReader(filePath))
            {
                string line;
                while ((line = reader.ReadLine()) != null)
                {
                    count++;
                    if (count > maxIterations)
                        yield break;

                    yield return JsonConvert.DeserializeObject<DrawingEntry>(line);
                }
            }
        }
    }
    public class Test_QuickDraw
    {
        public static bool[,] MakeFancyArray(DrawingEntry drawingEntry, int width, int height)
        {
            var points = drawingEntry.GetPointArray();
            bool[,] items = new bool[width, height];

            foreach (var stroke in points)
            {
                for (int i = 1; i < stroke.Count; i++)
                {
                    var x1 = (int)Math.Round(stroke[i-1].X);
                    var x2 = (int)Math.Round(stroke[i].X);
                    var y1 = (int)Math.Round(stroke[i-1].Y);
                    var y2 = (int)Math.Round(stroke[i].Y);

                    var dx = x2 - x1;
                    var dy = y2 - y1;
                    double m = (double)dy / (double)dx;

                    for (int x = x1; x <= x2; x++)
                    {
                        int y = (int)Math.Round(m * (x - x1) + y1);
                        items[x, y] = true;
                    }
                }
            }
            return items;
        }

        public static float[] MakeData(DrawingEntry drawingEntry, int width, int height)
        {
            drawingEntry.ResizeDrawing(width - 1, height - 1);

            bool[,] fancyArray = MakeFancyArray(drawingEntry, width, height);
            List<float> result = new(width * height);

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    result.Add(fancyArray[x, y] ? 1 : 0);
                }
            }

            return result.ToArray();

            if ((width * height) % 32 != 0)
                throw new Exception("Not 32 stuff! Stupid idiot!");

            int i = 0;
            byte b = 0;
            byte[] bytes = new byte[width * height / 8];

            for(int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    bool val = fancyArray[x, y];
                    b += (byte)((val ? 1 : 0) << i);
                    i++;

                    if (i == 8)
                    {
                        bytes[(y * width + x) / 8] = b;
                        i = 0;
                    }
                }
            }
        }

        public static void Run()
        {
            string filePath = "D:\\NN_Datasets";

            int width = 128;
            int height = 128;
            int images = 100;
            int categories = 4;

            var drawings = NdjsonParser.IterateFiles(filePath, images);

            foreach (var drawing in drawings)
            {
                foreach (var entry in drawing.Drawings)
                {
                    Console.WriteLine($"Key ID: {entry.KeyId}, Word: {entry.Word}, Recognized: {entry.Recognized}");
                    var points = entry.GetPointArray();
                    Console.WriteLine($"Total Points: {points.Count}");

                    entry.ResizeDrawing(1, 1);
                    Console.WriteLine("Drawing resized.");
                }
            }


            int addIndex = 0;
            int categoryIndex = 0;
            float[][] inputs = new float[images * categories][];
            float[][] desired = new float[images * categories][];
            foreach (var category in drawings)
            {
                addIndex = 0;
                foreach (var image in category.Drawings)
                {
                    inputs[addIndex * categoryIndex] = MakeData(image, width, height);
                    desired[addIndex * categoryIndex] = new float[4];
                    desired[addIndex * categoryIndex][categoryIndex] = 1;

                    addIndex++;
                }
                categoryIndex++;
            }

            var nnmodel = NetworkBuilder.Create()
                .Stack(new InputLayer(width * height))
                .Stack(new DenseLayer(8196, ActivationType.Sigmoid))
                .Stack(new DenseLayer(1024, ActivationType.Sigmoid))
                .Stack(new DenseLayer(256, ActivationType.Sigmoid))
                .Stack(new OutputLayer(4, ActivationType.Softmax))
                .Build();

            nnmodel.Summary();

            nnmodel.Train(inputs, desired, 10, 0.01f, 1, 1, 5);
            nnmodel.Save("D:\\baum.cool");
        }
    }
}
