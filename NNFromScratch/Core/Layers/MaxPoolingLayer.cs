
namespace NNFromScratch.Core.Layers;

public class MaxPoolingLayer : BaseLayer
{
    private int inputHeight, inputWidth, inputChannels;
    private int poolSize, stride;
    private int outputHeight, outputWidth;

    public MaxPoolingLayer(int inputHeight, int inputWidth, int inputChannels, int poolSize = 2, int stride = 2)
    {
        this.inputHeight = inputHeight;
        this.inputWidth = inputWidth;
        this.inputChannels = inputChannels;
        this.poolSize = poolSize;
        this.stride = stride;

        // Calculate the output dimensions after pooling
        outputHeight = (inputHeight - poolSize) / stride + 1;
        outputWidth = (inputWidth - poolSize) / stride + 1;

        Size = outputHeight * outputWidth * inputChannels;
    }

    public override void Initialize()
    {
        NeuronValues = new float[Size];
        Errors = new float[Size];
        Biases = new float[Size]; // Biases may not be used in pooling, but to align with BaseLayer
    }

    public override void FeedForward()
    {
        if (PreviousLayer == null)
            throw new Exception("Previous layer is not set for this pooling layer.");

        float[] inputValues = PreviousLayer.NeuronValues;
        float[,,] input3D = ConvertTo3D(inputValues, inputHeight, inputWidth, inputChannels);

        int index = 0;
        NeuronValues = new float[Size];

        // Apply Max Pooling operation
        for (int c = 0; c < inputChannels; c++)
        {
            for (int i = 0; i < outputHeight; i++)
            {
                for (int j = 0; j < outputWidth; j++)
                {
                    float maxVal = float.MinValue;

                    for (int pi = 0; pi < poolSize; pi++)
                    {
                        for (int pj = 0; pj < poolSize; pj++)
                        {
                            int startRow = i * stride + pi;
                            int startCol = j * stride + pj;
                            maxVal = Math.Max(maxVal, input3D[startRow, startCol, c]);
                        }
                    }

                    NeuronValues[index++] = maxVal;
                }
            }
        }
    }

    public override void Train(float[] desiredValues, float learningRate)
    {
        // Pooling layers typically do not need training as they perform a fixed operation.
        // However, we need to propagate errors back to the previous layer for backpropagation.
        float[] previousErrors = new float[PreviousLayer.NeuronValues.Length];
        float[,,] input3D = ConvertTo3D(PreviousLayer.NeuronValues, inputHeight, inputWidth, inputChannels);

        int index = 0;
        for (int c = 0; c < inputChannels; c++)
        {
            for (int i = 0; i < outputHeight; i++)
            {
                for (int j = 0; j < outputWidth; j++)
                {
                    float pooledValue = NeuronValues[index];
                    float error = Errors[index++];

                    // Distribute the error to the input neurons that contributed to the max value
                    for (int pi = 0; pi < poolSize; pi++)
                    {
                        for (int pj = 0; pj < poolSize; pj++)
                        {
                            int startRow = i * stride + pi;
                            int startCol = j * stride + pj;
                            if (input3D[startRow, startCol, c] == pooledValue)
                            {
                                previousErrors[startRow * inputWidth + startCol] += error;
                            }
                        }
                    }
                }
            }
        }

        // Propagate errors to the previous layer
        PreviousLayer.Errors = previousErrors;
    }

    public override void Summary()
    {
        Console.WriteLine($"MaxPooling Layer: {inputChannels} channels with pooling size {poolSize}x{poolSize} and stride {stride}");
    }

    public override void Save(BinaryWriter bw)
    {
        // Pooling layers generally do not need to save parameters since there are no weights or biases
    }

    public override void Load(BinaryReader br)
    {
        // Pooling layers generally do not need to load parameters
    }

    public override void InitializeCuda(int index)
    {
        // CUDA initialization not implemented in this sample
    }

    private float[,,] ConvertTo3D(float[] input, int height, int width, int channels)
    {
        float[,,] result = new float[height, width, channels];
        int index = 0;

        for (int c = 0; c < channels; c++)
        {
            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    result[i, j, c] = input[index++];
                }
            }
        }

        return result;
    }
}