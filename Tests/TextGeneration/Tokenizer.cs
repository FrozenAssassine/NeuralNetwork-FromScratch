using NNFromScratch.Helper;

namespace Tests.TextGeneration;

internal class Tokenizer
{
    public Dictionary<string, int> wordIndex;
    public string[] words;

    public Tokenizer(string text)
    {
        string[] words = text.Split(" ", StringSplitOptions.RemoveEmptyEntries);

        Dictionary<string, int> wordIndex = new Dictionary<string, int>(words.Length);
        for (int i = 0; i < text.Length; i++)
        {
            if (wordIndex.ContainsKey(words[i]))
                continue;

            wordIndex.Add(words[i], wordIndex.Count);
        }
        wordIndex.TrimExcess();

        Console.WriteLine("Number of unique words: " + wordIndex.Count);
    }

    public float[] MakeWordIndexTable()
    {
        float[] indexedText = new float[this.words.Length];

        for (int i = 0; i < this.words.Length; i++)
        {
            this.wordIndex.TryGetValue(this.words[i], out int wIndex);
            indexedText[i] = wIndex;
        }
        DatasetHelper.PrintItems(indexedText, 10);
        Console.WriteLine();

        return indexedText;
    }

}
