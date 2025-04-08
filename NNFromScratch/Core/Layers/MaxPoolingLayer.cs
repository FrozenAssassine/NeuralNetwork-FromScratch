using NNFromScratch.Helper;
using System;
using System.Collections.Generic;

namespace NNFromScratch.Core.Layers;

public class MaxPoolingLayer : BaseLayer
{
    private readonly int poolWidth;
    private readonly int poolHeight;
    private readonly int stride;
    private (int imageWidth, int imageHeight) inputSize;
    private (int width, int height) outputSize;
    private int[] maxIndices; // Store indices of max values for backpropagation
    private int numberFilters; // Number of feature maps

    public MaxPoolingLayer(int poolWidth, int poolHeight, (int imageWidth, int imageHeight) inputSize, int stride = 0)
    {
        this.poolWidth = poolWidth;
        this.poolHeight = poolHeight;
        this.stride = stride <= 0 ? poolWidth : stride; // Default stride to pool size if not specified
        this.inputSize = inputSize;
        this.outputSize = CalculateOutputSize(inputSize);
    }

    private (int width, int height) CalculateOutputSize((int imageWidth, int imageHeight) inputSize)
    {
        int outputHeight = (inputSize.imageHeight - poolHeight) / stride + 1;
        int outputWidth = (inputSize.imageWidth - poolWidth) / stride + 1;
        return (width: outputWidth, height: outputHeight);
    }

    private float[] GetPortion(float[] input, int startX, int startY)
    {
        float[] portion = new float[poolWidth * poolHeight];

        for (int y = 0; y < poolHeight; y++)
        {
            for (int x = 0; x < poolWidth; x++)
            {
                int inputIndex = (startY + y) * inputSize.imageWidth + (startX + x);
                int outputIndex = y * poolWidth + x;
                portion[outputIndex] = input[inputIndex];
            }
        }

        return portion;
    }

    public override void FeedForward()
    {
        // Calculate output dimensions
        int outputHeight = outputSize.height;
        int outputWidth = outputSize.width;

        // Initialize arrays
        NeuronValues = new float[Size];
        maxIndices = new int[Size]; // Store indices for backpropagation

        for (int f = 0; f < numberFilters; f++)
        {
            for (int y = 0; y < outputHeight; y++)
            {
                for (int x = 0; x < outputWidth; x++)
                {
                    int startX = x * stride;
                    int startY = y * stride;

                    // Get the input region to pool
                    float[] portion = GetInputPortion(f, startX, startY);

                    // Find max value and its index
                    var (maxIndex, maxValue) = MathHelper.GetMaxValIndex(portion);

                    // Store max value in output
                    int outputIndex = f * outputHeight * outputWidth + y * outputWidth + x;
                    NeuronValues[outputIndex] = maxValue;
                    maxIndices[outputIndex] = maxIndex;
                }
            }
        }
    }

    private float[] GetInputPortion(int filter, int startX, int startY)
    {
        float[] portion = new float[poolWidth * poolHeight];
        int filterOffset = filter * inputSize.imageWidth * inputSize.imageHeight;

        for (int y = 0; y < poolHeight; y++)
        {
            for (int x = 0; x < poolWidth; x++)
            {
                int inputIndex = filterOffset + (startY + y) * inputSize.imageWidth + (startX + x);
                int portionIndex = y * poolWidth + x;

                if (inputIndex < PreviousLayer.NeuronValues.Length)
                {
                    portion[portionIndex] = PreviousLayer.NeuronValues[inputIndex];
                }
            }
        }

        return portion;
    }

    public override void Train(float[] desiredValues, float learningRate)
    {
        FeedForward();

        // If this is the output layer, calculate errors
        if (NextLayer == null)
        {
            for (int i = 0; i < Size; i++)
            {
                Errors[i] = desiredValues[i] - NeuronValues[i];
            }
        }

        // Max pooling layer doesn't have weights to update

        // Propagate errors to previous layer
        PropagateErrorsToPreviousLayer();
    }

    private void PropagateErrorsToPreviousLayer()
    {
        if (this.PreviousLayer == null)
            return;

        // Reset previous layer errors
        Array.Clear(this.PreviousLayer.Errors, 0, this.PreviousLayer.Errors.Length);

        int outputHeight = outputSize.height;
        int outputWidth = outputSize.width;

        // For each filter
        for (int f = 0; f < numberFilters; f++)
        {
            int filterOffset = f * inputSize.imageWidth * inputSize.imageHeight;

            // For each position in the output
            for (int y = 0; y < outputHeight; y++)
            {
                for (int x = 0; x < outputWidth; x++)
                {
                    int outputIndex = f * outputHeight * outputWidth + y * outputWidth + x;
                    float error = Errors[outputIndex];

                    // Get the index of the max value that was selected during forward pass
                    int maxIndex = maxIndices[outputIndex];

                    // Convert 1D max index back to 2D coordinates within the pool window
                    int maxX = maxIndex % poolWidth;
                    int maxY = maxIndex / poolWidth;

                    // Calculate the index in the input
                    int startX = x * stride;
                    int startY = y * stride;
                    int inputIndex = filterOffset + (startY + maxY) * inputSize.imageWidth + (startX + maxX);

                    // Propagate error only to the max value position
                    if (inputIndex < PreviousLayer.Errors.Length)
                    {
                        PreviousLayer.Errors[inputIndex] += error;
                    }
                }
            }
        }
    }

    public override void Initialize(int inputCount, int outputCount)
    {
        if (PreviousLayer == null)
            throw new InvalidOperationException("PreviousLayer must be set before initializing MaxPoolingLayer.");

        // Determine the number of filters from the previous layer's output size
        numberFilters = PreviousLayer.Size / (inputSize.imageWidth * inputSize.imageHeight);

        // Recalculate output size based on final input dimensions
        outputSize = CalculateOutputSize(inputSize);

        // Calculate the total size of this layer's output
        Size = outputSize.width * outputSize.height * numberFilters;

        // Initialize arrays
        NeuronValues = new float[Size];
        Errors = new float[Size];
        maxIndices = new int[Size];

        // No weights or biases in max pooling layer
        Weights = new float[0];
        Biases = new float[0];
    }

    public override void Summary()
    {
        Console.WriteLine($"Max Pooling Layer: {poolWidth}x{poolHeight}, Input: {inputSize.imageWidth}x{inputSize.imageHeight}, Output: {outputSize.width}x{outputSize.height}, Filters: {numberFilters}");
    }

    public override void Save(BinaryWriter bw)
    {
        throw new NotImplementedException();
    }

    public override void Load(BinaryReader br)
    {
        throw new NotImplementedException();
    }

    public override void InitializeCuda(int index)
    {
        throw new NotImplementedException();
    }
}