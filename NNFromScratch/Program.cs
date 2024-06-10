using NNFromScratch.Core;

Run1.main(null);

//var res = MathHelper.MSE_Loss(new double[] { 1, 0, 0, 1 }, new double[] { 0, 0, 0, 0 });

//Console.WriteLine(res);

Layer input = new Layer(2, "Input");
Layer hidden1 = new Layer(4, "Hidden1");
Layer output = new Layer(1, "Output");

NeuralNetwork nn = new NeuralNetwork(input, new Layer[] {hidden1 }, output);

void test(float[] values)
{
    var res = nn.FeedForward(values);
    foreach (var pred in res)
    {
        Console.WriteLine(pred);
    }
}

for (int i = 0; i < 10000; i++)
{
    nn.Train2(new float[] { 0, 0 }, new float[] { 0 }, 1, 0.1f);
    nn.Train2(new float[] { 0, 1 }, new float[] { 1 }, 1, 0.1f);
    nn.Train2(new float[] { 1, 0 }, new float[] { 1 }, 1, 0.1f);
    nn.Train2(new float[] { 1, 1 }, new float[] { 0 }, 1, 0.1f);
}

test(new float[] { 0, 0 });
test(new float[] { 1, 0 });
test(new float[] { 0, 1 });
test(new float[] { 1, 1 });

Console.WriteLine("");

//SVGDrawingHelper svg = new SVGDrawingHelper();
//svg.Draw(nn);
/*
 using NNFromScratch.Core;

//var res = MathHelper.MSE_Loss(new double[] { 1, 0, 0, 1 }, new double[] { 0, 0, 0, 0 });

//Console.WriteLine(res);

Layer input = new Layer(2, "Input");
Layer hidden1 = new Layer(4, "Hidden1");
Layer output = new Layer(1, "Output");

NeuralNetwork nn = new NeuralNetwork(input, new Layer[] {hidden1 }, output);

void accuracy(float[][] inputs, float[][] desired, int epoch)
{
    //float accuracy = 0;

    //for (int i = 0; i < desired.Length; i++)
    //{
    //    accuracy += desired[i][0] - nn.FeedForward(inputs[i])[0];
    //}
    string res = "";
    for (int i = 0; i < 4; i++)
    {
        res += Math.Round(nn.FeedForward(inputs[i])[0], 4) + ":";
    }
    Console.WriteLine(res);
    //Console.WriteLine($"Epoch{epoch} Accuracy: " + (accuracy / 4) + ":" + accuracy);
}
/*
float predict(float[] values)
{
    string resstr = "";
    var res = nn.FeedForward(values);
    foreach (var pred in res)
    {
        resstr += pred + ":";
        //Console.WriteLine(pred);
    }
    Console.WriteLine(resstr);
}*/
/*
var inputs = new float[][] { new float[] { 0, 0 }, new float[] { 0, 1 }, new float[] { 1, 0 }, new float[] { 1, 1 } };
var outputs = new float[][] { new float[] { 0 }, new float[] { 1 }, new float[] { 1 }, new float[] { 0 } };

for (int i = 0; i < 17000; i++)
{
    nn.Train2(inputs[0], outputs[0], 50, 0.1f);
    nn.Train2(inputs[1], outputs[1], 50, 0.1f);
    nn.Train2(inputs[2], outputs[2], 50, 0.1f);
    nn.Train2(inputs[3], outputs[3], 50, 0.1f);

    if (i % 10 == 0)
    {
        accuracy(inputs, outputs, i);
    }
}

//SVGDrawingHelper svg = new SVGDrawingHelper();
//svg.Draw(nn);
*/