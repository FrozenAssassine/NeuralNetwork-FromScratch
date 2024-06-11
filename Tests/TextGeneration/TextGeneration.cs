using NNFromScratch.Core;
using NNFromScratch.Helper;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Tests.TextGeneration;

internal class TextGeneration
{
    public void Run()
    {
        var raw_text = LoadFile("D:\\testnn\\01 Harry Potter and the Sorcerers Stone.txt");
        var tokenizer = new Tokenizer(raw_text);

        NNModel model = new NNModel(new Layer[]
        {
            new Layer(tokenizer.words.Length),
            new Layer(512),
            new Layer(512),
            new Layer(1),
        });

        float[][] x = new float[tokenizer.words.Length][];
        float[][] y = new float[tokenizer.words.Length][];

        float[] wit = tokenizer.MakeWordIndexTable();

        for (int i = 0; i< tokenizer.words.Length; i+=5)
        {
            x[i] = new float[] { tokenizer.words.Skip(i).Take(5) };
        }

        model.Train(x, y, 10);

        model.Save("D:\\textgeneration1.cool");
    }


    public static string LoadFile(string path) 
    {
        string text = File.ReadAllText(path);
        text.Replace("\n", "").Replace("\r", "").ToLower();
        return text;
    }
}
