using NNFromScratch.Core.Layers;
using NNFromScratch.Core;

namespace Tests.NextSequencePrediction
{
    internal class Test_NextSequencePrediction
    {
        static Random random = new Random();
        static int numberOfLists = 5000;
        static int sequenceLength = 5;
        static int maxNumber = 100;

        public static void Run()
        {
            float[][] inputs = new float[numberOfLists][];
            float[][] desired = new float[numberOfLists][];

            for (int i = 0; i < numberOfLists; i++)
            {
                int randomStart = random.Next(maxNumber - sequenceLength);
                float[] randomSequence = new float[sequenceLength];
                for (int j = 0; j < sequenceLength; j++)
                {
                    randomSequence[j] = (randomStart + j) / (float)maxNumber;
                }

                inputs[i] = randomSequence;
                desired[i] = new float[1] { (randomStart + sequenceLength) / (float)maxNumber };
            }

            for (int i = 0; i < 100; i++)
            {
                Console.WriteLine("[" + string.Join("|", inputs[i]) + "] = " + desired[i][0]);
            }

            Console.WriteLine("Lists: " + inputs.Length);

            var network = NetworkBuilder.Create()
               .Stack(new InputLayer(sequenceLength))
               .Stack(new DenseLayer(512, ActivationType.Relu))
               .Stack(new OutputLayer(1, ActivationType.Softmax))
               .Build();

            network.Load("D:\\testnn\\nextsequencepred.cool");

            network.Train(inputs, desired, 100, 0.01f);
            network.Save("D:\\testnn\\nextsequencepred.cool");

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

                var vals = characteres.Select(x => float.Parse(x) / (float)maxNumber).ToArray();

                Console.WriteLine("[" + string.Join("|", vals) + "]");

                for (int i = 0; i < 50; i++)
                {
                    var prediction = network.Predict(vals, false).First();

                    Console.Write($" = {i + 1}: {prediction * (float)maxNumber} ");

                    vals = vals.Skip(1).Append(prediction).ToArray();
                    Console.WriteLine("Updated input: [" + string.Join("|", vals) + "]");
                }
            }
        }
    }
}
