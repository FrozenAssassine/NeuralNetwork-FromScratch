using NNFromScratch.Core;
using NNFromScratch.Helper;

var res = MathHelper.MSE_Loss(new double[] { 1, 0, 0, 1 }, new double[] { 0, 0, 0, 0 });

Console.WriteLine(res);

NeuronalNetwork nn = new NeuronalNetwork(10, 30, 8);
nn.FeedForward(new double[] { 100, 20, 10});

SVGDrawingHelper svg = new SVGDrawingHelper();
svg.Draw(nn);