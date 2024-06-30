using NNFromScratch.Core.Layers;
using NNFromScratch.Core;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Tests.TextGeneration
{
    internal class Test_TextGeneration
    {
        const string datasetPath = "D:\\testnn\\text_generation\\train.txt"; // Path to your text dataset
        const int SequenceLength = 100; // Length of each input sequence
        const int VocabularySize = 256; // ASCII characters

        public static void Run()
        {
            // Load and preprocess the text data
            string text = File.ReadAllText(datasetPath);
            int textLength = text.Length;

            // Create input and target sequences
            float[][] inputs = new float[textLength - SequenceLength][];
            float[][] targets = new float[textLength - SequenceLength][];

            for (int i = 0; i < textLength - SequenceLength; i++)
            {
                inputs[i] = new float[SequenceLength];
                targets[i] = new float[VocabularySize];

                for (int j = 0; j < SequenceLength; j++)
                {
                    inputs[i][j] = text[i + j];
                }

                targets[i][text[i + SequenceLength]] = 1;
            }


            // Build the network
            var network = NetworkBuilder.Create()
                .Stack(new InputLayer(SequenceLength))
                .Stack(new NeuronLayer(1024, ActivationType.Tanh))
                .Stack(new NeuronLayer(512, ActivationType.Tanh))
                .Stack(new OutputLayer(VocabularySize, ActivationType.Softmax))
                .Build();

            network.Load("D:\\textgeneration.cool");

            // Train the network
            network.Train(inputs, targets, 20, 0.1f); // Train for 10 epochs

            // Save the trained model
            Console.WriteLine("Press enter to Save");
            Console.ReadLine();
            network.Save("D:\\textgeneration.cool");

            // Generate text
            Console.WriteLine("Enter seed text:");
            string seedText = Console.ReadLine();
            string generatedText = GenerateText(network, seedText, 100);
            Console.WriteLine("Generated Text:");
            Console.WriteLine(generatedText);
        }

        private static string GenerateText(NNModel network, string seedText, int length)
        {
            StringBuilder generated = new StringBuilder(seedText);
            float[] input = new float[SequenceLength];

            // Initialize input with seed text
            for (int i = 0; i < seedText.Length && i < SequenceLength; i++)
            {
                input[i] = seedText[i];
            }

            for (int i = 0; i < length; i++)
            {
                // Get the prediction from the network
                float[] output = network.Predict(input);
                int predictedChar = ArgMax(output);

                //Console.WriteLine("Pred: " + predictedChar + ":END");
                generated.Append((char)predictedChar);

                // Shift input sequence and add the predicted character
                for (int j = 0; j < SequenceLength - 1; j++)
                {
                    input[j] = input[j + 1];
                }
                input[SequenceLength - 1] = predictedChar;
            }

            return generated.ToString();
        }

        private static int ArgMax(float[] array)
        {
            int maxIndex = 0;
            float maxValue = array[1];
            for (int i = 1; i < array.Length; i++)
            {
                if (array[i] > maxValue)
                {
                    maxIndex = i;
                    maxValue = array[i];
                }
            }
            return maxIndex;
        }
    }
}
