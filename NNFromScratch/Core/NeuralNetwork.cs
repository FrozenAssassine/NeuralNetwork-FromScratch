using NNFromScratch.Helper;

namespace NNFromScratch.Core;

internal class NeuralNetwork
{
    public readonly Layer inputLayer;
    public readonly Layer[] hiddenLayers;
    public readonly Layer outputLayer;

    private int CountNeurons(Layer inputs, Layer[] hidden, Layer outputs)
    {
        int neurons = inputs.Size * hidden[0].Size;

        for(int i = 0; i<hidden.Length - 1; i++)
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
        inputLayer.Initialize(null, hidden[0]);
        for(int i = 0; i<hidden.Length; i++)
        {
            if (i == 0)
                hidden[i].Initialize(inputLayer, outputLayer);
            else
                hidden[i].Initialize(hidden[i - 1], i > hidden.Length ? outputLayer : hidden[i]);
        }
        outputLayer.Initialize(hidden[hidden.Length > 1 ? hidden.Length - 1 : 0], null);
    }

    /*public void Train(float[] inputs, float[] desiredOutputs, int epochs, float learningRate)
    {
        //for(int e = 0; e < epochs; e++)
        //{
        //    float[] res = FeedForward(inputs);
        //    for(int i = 0; i < outputLayer.NeuronValues.Length; i++)
        //    {
        //        outputLayer.Errors[i] = res[i] - desiredOutputs[i];
        //        for(int j = 0; j < outputLayer.PreviousLayer.NeuronValues.Length; j++)
        //        {
        //            outputLayer.Weight[outputLayer.PreviousLayer.NeuronValues.Length * i + j] = learningRate * outputLayer.Errors[i] * MathHelper.SigmoidDerivative(outputLayer.NeuronValues[i]);
        //        }
        //    }
        //}

        for (int e = 0; e < epochs; e++)
        {
            // Perform feedforward pass to get the network's output
            float[] res = FeedForward(inputs);

            // Calculate errors for the output layer
            for (int i = 0; i < outputLayer.NeuronValues.Length; i++)
            {
                outputLayer.Errors[i] = res[i] - desiredOutputs[i];

                // Update weights for the output layer
                for (int j = 0; j < outputLayer.PreviousLayer.NeuronValues.Length; j++)
                {
                    int weightIndex = outputLayer.PreviousLayer.NeuronValues.Length * i + j;
                    outputLayer.Weight[weightIndex] -= learningRate * outputLayer.Errors[i] * MathHelper.SigmoidDerivative(outputLayer.NeuronValues[i]) * outputLayer.PreviousLayer.NeuronValues[j];
                }

                // Update biases for the output layer
                outputLayer.Biases[i] -= learningRate * outputLayer.Errors[i] * MathHelper.SigmoidDerivative(outputLayer.NeuronValues[i]);
            }
        }

    }*/

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
                    outputLayer.Weight[weightIndex] += learningRate * outputLayer.Errors[i] * MathHelper.SigmoidDerivative(outputLayer.NeuronValues[i]) * outputLayer.PreviousLayer.NeuronValues[j];
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
                    error += outputLayer.Errors[j] * outputLayer.Weight[weightIndex];
                }
                outputLayer.PreviousLayer.Errors[i] = error * MathHelper.SigmoidDerivative(outputLayer.PreviousLayer.NeuronValues[i]);
            }

            // Update weights for the hidden layer
            for (int i = 0; i < outputLayer.PreviousLayer.NeuronValues.Length; i++)
            {
                for (int j = 0; j < inputs.Length; j++)
                {
                    int weightIndex = inputs.Length * i + j;
                    outputLayer.PreviousLayer.Weight[weightIndex] += learningRate * outputLayer.PreviousLayer.Errors[i] * inputs[j];
                }

                // Update biases for the hidden layer
                outputLayer.PreviousLayer.Biases[i] += learningRate * outputLayer.PreviousLayer.Errors[i];
            }
        }
    }

    
}
