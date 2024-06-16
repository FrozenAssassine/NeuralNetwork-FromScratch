using NNFromScratch.Helper;
using static System.Net.Mime.MediaTypeNames;

namespace NNFromScratch.Core;

internal class NeuralNetwork
{
    public readonly Layer inputLayer;
    public readonly Layer[] hiddenLayers;
    public readonly Layer outputLayer;

    public NeuralNetwork(Layer inputs, Layer[] hidden, Layer outputs)
    {
        this.inputLayer = inputs;
        this.hiddenLayers = hidden;
        this.outputLayer = outputs;

        //initialize the layers 
        inputLayer.Initialize(null);
        for (int i = 0; i < hidden.Length; i++)
        {
            if (i == 0)
                hidden[i].Initialize(inputLayer);
            else
                hidden[i].Initialize(hidden[i - 1]);
        }
        outputLayer.Initialize(hidden[hidden.Length > 1 ? hidden.Length - 1 : 0]);
    }

    private void Train_CPU(float[] inputs, float[] desiredOutputs, float learningRate)
    {
        // Perform feedforward pass to get the network's output
        float[] res = FeedForward_CPU(inputs);

        // Calculate errors for the output layer
        Parallel.For(0, outputLayer.Size, (i) =>
        {
            outputLayer.Errors[i] = desiredOutputs[i] - res[i];
        });

        // Update weights and biases for the output layer
        Parallel.For(0, outputLayer.Size, (i) =>
        {
            for (int j = 0; j < outputLayer.PreviousLayer.Size; j++)
            {
                int weightIndex = i * outputLayer.PreviousLayer.Size + j;
                outputLayer.Weights[weightIndex] += learningRate * outputLayer.Errors[i] * MathHelper.SigmoidDerivative(outputLayer.NeuronValues[i]) * outputLayer.PreviousLayer.NeuronValues[j];
            }
            outputLayer.Biases[i] += learningRate * outputLayer.Errors[i] * MathHelper.SigmoidDerivative(outputLayer.NeuronValues[i]);
        });

        // Backpropagate the errors to the hidden layers
        for (int h = hiddenLayers.Length - 1; h >= 0; h--)
        {
            //int h = hiddenLayers.Length - 1 - index;
            Layer currentLayer = hiddenLayers[h];
            Layer nextLayer = (h == hiddenLayers.Length - 1) ? outputLayer : hiddenLayers[h + 1];
            Layer previousLayer = (h == 0) ? inputLayer : hiddenLayers[h - 1];

            float error = 0.0f;
            
            Parallel.For(0, currentLayer.Size, (i) =>
            {
                error = 0.0f;
                //calculate and update error:
                for (int j = 0; j < nextLayer.Size; j++)
                {
                    error += (nextLayer.Errors[j] * nextLayer.Weights[j * currentLayer.Size + i]);
                }
                currentLayer.Errors[i] = error * MathHelper.SigmoidDerivative(currentLayer.NeuronValues[i]);

                //update biases and weights:
                for (int j = 0; j < previousLayer.Size; j++)
                {
                    currentLayer.Weights[i * previousLayer.Size + j] +=
                        learningRate * currentLayer.Errors[i] * previousLayer.NeuronValues[j];
                }
                currentLayer.Biases[i] += learningRate * currentLayer.Errors[i];
            });
        }
    }
    private float[] FeedForward_CPU(float[] data)
    {
        if (data.Length != inputLayer.Size)
            throw new Exception("Input size is not the same as the number of layers");

        for (int i = 0; i < data.Length; i++)
        {
            inputLayer.NeuronValues[i] = data[i];
        }

        //calculate neuron values for hidden layer:
        foreach (var hidden in hiddenLayers)
        {
            for (int i = 0; i < hidden.Size; i++)
            {
                float sum = 0.0f;
                for (int j = 0; j < hidden.PreviousLayer.Size; j++)
                {
                    sum += hidden.PreviousLayer.NeuronValues[j] * hidden.Weights[i * hidden.PreviousLayer.Size + j];
                }
                hidden.NeuronValues[i] = MathHelper.Sigmoid(sum + hidden.Biases[i]);
            }
        }

        //Compute neuron values for output layer
        Parallel.For(0, outputLayer.Size, (i) =>
        {
            float sum = 0.0f;
            for (int j = 0; j < outputLayer.PreviousLayer.Size; j++)
            {
                int weightIndex = i * outputLayer.PreviousLayer.Size + j;
                sum += outputLayer.PreviousLayer.NeuronValues[j] * outputLayer.Weights[weightIndex];
            }
            outputLayer.NeuronValues[i] = MathHelper.Sigmoid(sum + outputLayer.Biases[i]);
        });

        return outputLayer.NeuronValues;
    }
    private float[] FeedForward_GPU(float[] data)
    {
        float[] prediction = new float[outputLayer.Size];
        CudaAccel.Predict(data, prediction, inputLayer.Size);
        return prediction;
    }

    public float[] FeedForward(float[] data, bool useCuda)
    {
        if (useCuda)
            return FeedForward_GPU(data);
        return FeedForward_CPU(data);
    }

    public void Train(float[] inputs, float[] desiredOutputs, float learningRate, bool useCuda)
    {
        if (useCuda)
            CudaAccel.Train(inputs, desiredOutputs, inputs.Length, learningRate);
        else
            Train_CPU(inputs, desiredOutputs, learningRate);
    }

    public void Save(Stream stream)
    {
        //todo:
        var layer = new List<Layer>();
        layer.AddRange(hiddenLayers);
        layer.Add(outputLayer);

        BinaryWriter bw = new BinaryWriter(stream);
        foreach(var l in layer)
        {
            l.Save(bw);
        }
        bw.Dispose();
    }

    public void Load(Stream stream)
    {
        //todo:
        var layer = new List<Layer>();
        layer.AddRange(hiddenLayers);
        layer.Add(outputLayer);

        BinaryReader br = new BinaryReader(stream);
        foreach (var l in layer)
        {
            l.Load(br);
        }
        br.Dispose();
    }

    public void Summary()
    {
        Console.WriteLine(new string('-', 50));
        inputLayer.Summary();
        foreach (var hidden in hiddenLayers)
        {
            hidden.Summary();
        }
        outputLayer.Summary();
        Console.WriteLine(new string('=', 50));
    }
}
