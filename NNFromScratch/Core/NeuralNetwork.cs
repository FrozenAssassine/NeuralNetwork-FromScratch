﻿using NNFromScratch.Helper;
using System.Linq;

namespace NNFromScratch.Core;

internal class NeuralNetwork
{
    public readonly Layer inputLayer;
    public readonly Layer[] hiddenLayers;
    public readonly Layer outputLayer;

    private int CountNeurons(Layer inputs, Layer[] hidden, Layer outputs)
    {
        int neurons = inputs.Size * hidden[0].Size;

        for (int i = 0; i < hidden.Length - 1; i++)
        {
            neurons += hidden[i].Size * hidden[i - 1].Size;
        }

        neurons += hidden[hidden.Length > 1 ? hidden.Length - 1 : 0].Size * outputs.Size;
        return neurons;
    }

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

    public void Train(float[] inputs, float[] desiredOutputs, int epochs, float learningRate)
    {
        for (int e = 0; e < epochs; e++)
        {
            // Perform feedforward pass to get the networks output
            float[] res = FeedForward(inputs);

            // Calculate errors for the output layer
            for (int i = 0; i < outputLayer.NeuronValues.Length; i++)
            {
                outputLayer.Errors[i] = res[i] - desiredOutputs[i];

                // Update weights for the output layer
                for (int j = 0; j < outputLayer.PreviousLayer.NeuronValues.Length; j++)
                {
                    int weightIndex = outputLayer.PreviousLayer.NeuronValues.Length * i + j;
                    outputLayer.Weights[weightIndex] += learningRate * outputLayer.Errors[i] * MathHelper.SigmoidDerivative(outputLayer.NeuronValues[i]) * outputLayer.PreviousLayer.NeuronValues[j];
                }

                // Update biases for the output layer
                outputLayer.Biases[i] += learningRate * outputLayer.Errors[i] * MathHelper.SigmoidDerivative(outputLayer.NeuronValues[i]);
            }

            // Calculate the errors fro the hidden layer
            for (int i = 0; i < outputLayer.PreviousLayer.NeuronValues.Length; i++)
            {
                float error = 0.0f;
                for (int j = 0; j < outputLayer.NeuronValues.Length; j++)
                {
                    int weightIndex = outputLayer.PreviousLayer.NeuronValues.Length * j + i;
                    error += outputLayer.Errors[j] * outputLayer.Weights[weightIndex];
                }
                outputLayer.PreviousLayer.Errors[i] = error * MathHelper.SigmoidDerivative(outputLayer.PreviousLayer.NeuronValues[i]);
            }

            // Update weights for the hidden layer
            for (int i = 0; i < outputLayer.PreviousLayer.NeuronValues.Length; i++)
            {
                for (int j = 0; j < inputs.Length; j++)
                {
                    int weightIndex = inputs.Length * i + j;
                    outputLayer.PreviousLayer.Weights[weightIndex] += learningRate * outputLayer.PreviousLayer.Errors[i] * inputs[j];
                }

                // Update biases for the hidden layer
                outputLayer.PreviousLayer.Biases[i] += learningRate * outputLayer.PreviousLayer.Errors[i];
            }
        }
    }

    public void Train2(float[] inputs, float[] desiredOutputs, int epochs, float learningRate)
    {
        for (int e = 0; e < epochs; e++)
        {
            // Perform feedforward pass to get the network's output
            float[] res = FeedForward(inputs);

            // Calculate errors for the output layer
            Parallel.For(0, outputLayer.NeuronValues.Length, (i) =>
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
            Parallel.For(0, hiddenLayers.Length, index =>
            {
                int h = hiddenLayers.Length - 1 - index;

                Layer currentLayer = hiddenLayers[h];
                Layer nextLayer = (h == hiddenLayers.Length - 1) ? outputLayer : hiddenLayers[h + 1];

                for (int i = 0; i < currentLayer.Size; i++)
                {
                    float error = 0.0f;
                    for (int j = 0; j < nextLayer.Size; j++)
                    {
                        int weightIndex = j * currentLayer.Size + i;
                        error += nextLayer.Errors[j] * nextLayer.Weights[weightIndex];
                    }
                    currentLayer.Errors[i] = error * MathHelper.SigmoidDerivative(currentLayer.NeuronValues[i]);
                }
            });

            // Update weights and biases for hidden layers
            Parallel.For(0, hiddenLayers.Length, index =>
            {
                int h = hiddenLayers.Length - 1 - index;

                Layer currentLayer = hiddenLayers[h];
                Layer previousLayer = (h == 0) ? inputLayer : hiddenLayers[h - 1];

                for (int i = 0; i < currentLayer.Size; i++)
                {
                    for (int j = 0; j < previousLayer.Size; j++)
                    {
                        int weightIndex = i * previousLayer.Size + j;
                        currentLayer.Weights[weightIndex] += learningRate * currentLayer.Errors[i] * previousLayer.NeuronValues[j];
                    }
                    currentLayer.Biases[i] += learningRate * currentLayer.Errors[i];
                }
            });
        }
    }

    public float[] FeedForward(float[] data)
    {
        if (data.Length != inputLayer.Size)
            throw new Exception("Input size is not the same as the number of layers");

        //fill the input neurons with its corresponding data
        Parallel.For(0, data.Length, (i) => {
            inputLayer.NeuronValues[i] = data[i];
        });


        foreach (var hidden in hiddenLayers)
        {
            Parallel.For(0, hidden.Size, (i) =>
            {
                float sum = 0.0f;
                for (int j = 0; j < hidden.PreviousLayer.Size; j++)
                {
                    int weightIndex = i * hidden.PreviousLayer.Size + j;
                    sum += hidden.PreviousLayer.NeuronValues[j] * hidden.Weights[weightIndex];
                }
                sum += hidden.Biases[i];
                hidden.NeuronValues[i] = MathHelper.Sigmoid(sum);
            });
        }

        // Compute neuron values for output layer
        Parallel.For(0, outputLayer.Size, (i) =>
        {
            float sum = 0.0f;
            for (int j = 0; j < outputLayer.PreviousLayer.Size; j++)
            {
                int weightIndex = i * outputLayer.PreviousLayer.Size + j;
                sum += outputLayer.PreviousLayer.NeuronValues[j] * outputLayer.Weights[weightIndex];
            }
            sum += outputLayer.Biases[i];
            outputLayer.NeuronValues[i] = MathHelper.Sigmoid(sum);
        });
        return outputLayer.NeuronValues;
    }

    public void SaveWeights()
    {

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
}
