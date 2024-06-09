using NNFromScratch.Core;
using NNFromScratch.Helper;

//var res = MathHelper.MSE_Loss(new double[] { 1, 0, 0, 1 }, new double[] { 0, 0, 0, 0 });

//Console.WriteLine(res);

Layer input = new Layer(4, "Input");
Layer hidden1 = new Layer(2, "Hidden1");
Layer output = new Layer(1, "Output");

NeuralNetwork nn = new NeuralNetwork(input, new Layer[] {hidden1 }, output);

nn.Train(new float[] { 00, 01, 10, 11 }, new float[] { 1,0,0,1 }, 10_000, 0.1f);

var res = nn.FeedForward(new float[] { 1, 0 });
foreach(var pred in res)
{
    Console.WriteLine(pred);
}

Console.WriteLine("");

SVGDrawingHelper svg = new SVGDrawingHelper();
svg.Draw(nn);