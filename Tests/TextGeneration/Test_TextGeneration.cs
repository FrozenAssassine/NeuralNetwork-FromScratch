using NNFromScratch.Core.Layers;
using NNFromScratch.Core;
using System.Text;
using SixLabors.ImageSharp;
using NNFromScratch.Helper;

namespace Tests.TextGeneration
{
    internal class Test_TextGeneration
    {
        const string datasetPath = "D:\\testnn\\01 Harry Potter and the Sorcerers Stone.txt";
        const int WordsLength = 8;
        const int MaxWordIndexBit = 8; //the number of bits for the arrays
        const int Length = 1000;
        public static void Run()
        {
            string text = File.ReadAllText(datasetPath);
            text = text.Remove(Length);
            text = CleanUp(text);

            var words = text.Split(new char[] { '.', ' ', ':' }, StringSplitOptions.RemoveEmptyEntries);
            int textLength = words.Length;
            var tokenIndexTable = Tokenize(words);
            int inputSize = textLength - WordsLength;

            Console.WriteLine("Number of unique words: " + tokenIndexTable.Count);
            Console.WriteLine("Training size: " + inputSize);

            float[][] inputs = new float[inputSize][];
            float[][] targets = new float[inputSize][];

            for (int i = 0; i < inputSize; i++)
            {
                float[] inputItem = new float[WordsLength * MaxWordIndexBit];
                float[] targetItem = new float[MaxWordIndexBit];

                for (int j = 0; j < WordsLength; j++)
                {
                    if (!tokenIndexTable.TryGetValue(words[i + j], out int val))
                        continue;

                    var bin = GetBinary(val);
                    Array.Copy(bin, 0, inputItem, j * MaxWordIndexBit, MaxWordIndexBit);
                }

                inputs[i] = inputItem;

                if (tokenIndexTable.TryGetValue(words[i + WordsLength], out int targetVal))
                {
                    targetItem = GetBinary(targetVal);
                }
                 
                targets[i] = targetItem;
            }

            // Build the network
            var network = NetworkBuilder.Create()
                .Stack(new InputLayer(WordsLength * MaxWordIndexBit))
                .Stack(new RNNLayer(WordsLength * MaxWordIndexBit, ))
                .Stack(new DenseLayer(WordsLength * MaxWordIndexBit, ActivationType.Sigmoid))
                .Stack(new OutputLayer(MaxWordIndexBit, ActivationType.Sigmoid))
                .Build(false);

            //network.Load("D:\\textgeneration.cool");

            Predict(network, tokenIndexTable, "A sick man's appetite", 10);

            // Train the network
            network.Train(inputs, targets, 500, 0.1f, 1000);

            //evaluation:
            var X_test = inputs.Take(1000).ToArray();
            var y_test = targets.Take(1000).ToArray();
            int correct = 0;
            for (int i = 0; i < X_test.Length; i++)
            {
                var ff = network.nn.FeedForward_CPU(X_test[i]);

                Console.Write(string.Join(", ", y_test[i]) + " -> ");
                Console.WriteLine(string.Join(", ", ff.Select(x => x < 0.7 ? 0 : 1)));

                if (BinaryToInt(y_test[i]) == BinaryToInt(ff))
                    correct++;
            }

            Console.WriteLine($"Evaluation: {correct}/{X_test.Length}");

            // Save the trained model
            Console.WriteLine("Press enter to Save");
            Console.ReadLine();
            network.Save("D:\\textgeneration.cool");

            Predict(network, tokenIndexTable, "A sick man's appetite", 10);
        }

        private static string CleanUp(string text)
        {
            return text
                .Replace("\n", "")
                .Replace("\r", "")
                .Replace(",", "")
                .Replace(":", "")
                .Replace(";", "")
                .Replace("'ll", " will")
                .Replace("'s", " is")
                .Replace("n't", "not");
        }

        private static void Predict(NNModel model, Dictionary<string, int> tokenIndexTable, string seed, int words = 5)
        {
            // Example input sequence
            string[] inputWords = seed.Split(" ");

            // Prepare the input sequence
            List<float> inputSequence = PrepareInputSequence(inputWords, tokenIndexTable);
            StringBuilder sentence = new StringBuilder();
            sentence.Append(seed + " ");
            for (int word = 0; word < words; word++)
            {
                //remove word arrays from the beginning because of limited length
                if(inputSequence.Count > WordsLength)
                {
                    inputSequence.RemoveRange(0, WordsLength);
                }

                float[] predictedBinary = model.Predict(inputSequence.ToArray());

                //get the next word from the prediction:
                var convertRes = BinaryToInt(predictedBinary);

                if(convertRes.number > tokenIndexTable.Count)
                {
                    inputSequence.AddRange(new float[] {0,0,0,0,0,0,0,0});
                    continue;
                }

                string nextWord = tokenIndexTable.Keys.ElementAt(convertRes.number);

                inputSequence.AddRange(convertRes.binaryArray);

                sentence.Append(nextWord + " ");
            }

            Console.WriteLine(sentence.ToString());
        }

        private static List<float> PrepareInputSequence(string[] words, Dictionary<string, int> tokenIndexTable)
        {
            List<float> inputItem = new List<float>(WordsLength * 8);

            for (int j = 0; j < WordsLength; j++)
            {
                if (j >= words.Length)
                    break;

                if (!tokenIndexTable.TryGetValue(words[j], out int val))
                    val = 0;

                var bin = GetBinary(val);
                inputItem.AddRange(bin);
            }

            return inputItem;
        }

        private static (int number, float[] binaryArray) BinaryToInt(float[] binaryArray)
        {
            //make a real binary value from all values in the array that are greater than 0.5
            float[] binaryValues = new float[binaryArray.Length];
            for (int i = 0; i < binaryArray.Length; i++)
            {
                binaryValues[i] = binaryArray[i] > 0.6 ? 1 : 0;
            }

            string binaryString = string.Join("", binaryValues.Select(b => ((int)b).ToString()));
            return (Convert.ToInt32(binaryString, 2), binaryArray);
        }

        private static float[] GetBinary(int number)
        {
            //todo make this better and faster
            string binaryString = Convert.ToString(number, 2);

            var newStr = new string('0', MaxWordIndexBit - binaryString.Length) + binaryString;

            float[] binaryArray = new float[newStr.Length];

            for (int i = 0; i < newStr.Length; i++)
            {
                binaryArray[i] = float.Parse(newStr[i].ToString());
            }

            return binaryArray;
        }

        /*
        private static string GenerateText(NNModel network, string seedText, int length)
        {
            StringBuilder generated = new StringBuilder(seedText);
            float[] input = new float[SequenceLength];

            for(int i = 0; i<seedText.Length; i++)
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
        */
        private static Dictionary<string, int> Tokenize(string[] words)
        {
            Dictionary<string, int> tokenLookupTable = new();
            tokenLookupTable.Add(" ", 0);

            foreach(var word in words)
            {
                if (!tokenLookupTable.ContainsKey(word))
                    tokenLookupTable.Add(word, tokenLookupTable.Count - 1);
            }

            return tokenLookupTable;
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
