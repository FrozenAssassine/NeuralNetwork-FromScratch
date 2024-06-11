using NNFromScratch.Core;
using NNFromScratch.Helper;

//var res = MathHelper.MSE_Loss(new double[] { 1, 0, 0, 1 }, new double[] { 0, 0, 0, 0 });

//Console.WriteLine(res);

Layer input = new Layer(2, "Input");
Layer hidden1 = new Layer(4, "Hidden1");
Layer output = new Layer(1, "Output");

NeuralNetwork nn = new NeuralNetwork(input, new Layer[] {hidden1 }, output);

var test = new float[] { 0 };

for (int i = 0; i < 10000; i++)
{
    nn.Train(new float[] { 0, 0 }, new float[] { 0 }, 200, 0.01f);
    nn.Train(new float[] { 0, 1 }, new float[] { 1 }, 200, 0.01f);
    nn.Train(new float[] { 1, 0 }, new float[] { 1 }, 200, 0.01f);
    nn.Train(new float[] { 1, 1 }, new float[] { 0 }, 200, 0.01f);

    var res = nn.FeedForward(new float[] { 1, 1 });
    for(int j = 0; j<res.Length; j++)
    {
        Console.WriteLine(res[j] + ":" + test[j]);// Math.Abs(test[j] - res[j]));
    }
}

var res2 = nn.FeedForward(new float[] { 1, 1 });

foreach(var pred in res2)
{
    Console.WriteLine(pred);
}

Console.WriteLine("");

SVGDrawingHelper svg = new SVGDrawingHelper();
svg.Draw(nn);