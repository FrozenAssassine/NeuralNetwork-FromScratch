using NNFromScratch.Core.Layers;
using NNFromScratch.Core;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http.Headers;
using System.Text;
using System.Threading.Tasks;

namespace Tests.SortedListCheck
{
    internal class Test_SortedListCheck
    {

        static bool IsSorted(float[] list)
        {
            for (int i = 1; i < list.Length; i++)
            {
                if (list[i] < list[i - 1])
                    return false;
            }

            return true;
        }

        public static void Run()
        {
            Random random = new Random();
            int numberOfLists = 10000;
            int itemsPerList = 3;
            int maxNumber = 50;

            float[][] inputs = new float[numberOfLists][];
            float[][] desired = new float[numberOfLists][];

            for (int i = 0; i < numberOfLists; i++)
            {
                var vals = Enumerable.Range(0, itemsPerList).Select(_ => random.Next(1, 101) / 100.0f).ToArray();

                inputs[i] = vals;
                desired[i] = new float[] { IsSorted(vals) ? 1 : 0 };
            }

            Console.WriteLine("Lists: " + inputs.Length);

            var network = NetworkBuilder.Create()
               .Stack(new InputLayer(itemsPerList))
               .Stack(new DenseLayer(512, ActivationType.Sigmoid))
               .Stack(new OutputLayer(1, ActivationType.Softmax))
               .Build();

            network.Load("D:\\testnn\\sortedlistcheck.cool");

            network.Train(inputs, desired, 30, 0.01f);
            network.Save("D:\\testnn\\sortedlistcheck.cool");

            var randomIndices = Enumerable.Range(0, 1000)
                                          .Select(_ => random.Next(0, numberOfLists))
                                          .Distinct()
                                          .ToList();

            var randInputs = randomIndices.Select(index => inputs[index]);
            var randDesired = randomIndices.Select(index => desired[index]);

            network.Evaluate(randInputs.ToArray(), randDesired.ToArray(), false);

            while (true)
            {
                Console.WriteLine("Enter test array: ");

                var data = Console.ReadLine();
                if (data == null)
                    return;

                var characteres = data.Split(',');

                var vals = characteres.Select(x => float.Parse(x) / maxNumber).ToArray();
                var prediction = network.Predict(vals, false).First();

                Console.WriteLine((prediction > 0.7f ? "SORTED" : "NOT") + " [" + string.Join("|", vals) + "] = " + prediction);


            }
        }
    }
}
