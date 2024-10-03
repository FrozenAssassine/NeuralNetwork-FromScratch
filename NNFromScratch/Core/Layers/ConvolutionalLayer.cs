
namespace NNFromScratch.Core.Layers;

public class ConvolutionalLayer : BaseLayer
{
    private int inputHeight, inputWidth, inputChannels;
    private int filterSize, numFilters, stride;
    private float[,,,] filters; // 4D array: [numFilters, filterHeight, filterWidth, inputChannels]
    private float[] biases;

    public ConvolutionalLayer(int inputHeight, int inputWidth, int inputChannels, int filterSize, int numFilters, int stride = 1)
    {
        this.inputHeight = inputHeight;
        this.inputWidth = inputWidth;
        this.inputChannels = inputChannels;
        this.filterSize = filterSize;
        this.numFilters = numFilters;
        this.stride = stride;

        // Initialize filters and biases
        Initialize();
    }

    public override void Initialize()
    {
        Random rand = new Random();
        filters = new float[numFilters, filterSize, filterSize, inputChannels];
        biases = new float[numFilters];

        // Initialize filters with random values
        for (int f = 0; f < numFilters; f++)
        {
            for (int c = 0; c < inputChannels; c++)
            {
                for (int i = 0; i < filterSize; i++)
                {
                    for (int j = 0; j < filterSize; j++)
                    {
                        filters[f, i, j, c] = (float)(rand.NextDouble() * 2 - 1); // Range [-1, 1]
                    }
                }
            }
            biases[f] = 0.0f;
        }
    }

    public override void FeedForward()
    {
        if (PreviousLayer == null)
            throw new Exception("Previous layer is not set for this convolutional layer.");

        float[] inputValues = PreviousLayer.NeuronValues;
        float[,,] input3D = ConvertTo3D(inputValues, inputHeight, inputWidth, inputChannels);

        int outputHeight = (inputHeight - filterSize) / stride + 1;
        int outputWidth = (inputWidth - filterSize) / stride + 1;
        NeuronValues = new float[outputHeight * outputWidth * numFilters];

        int index = 0;

        // Perform the convolution
        for (int f = 0; f < numFilters; f++)
        {
            for (int i = 0; i < outputHeight; i++)
            {
                for (int j = 0; j < outputWidth; j++)
                {
                    float value = 0.0f;

                    // Apply filter on the input 3D array
                    for (int c = 0; c < inputChannels; c++)
                    {
                        for (int fi = 0; fi < filterSize; fi++)
                        {
                            for (int fj = 0; fj < filterSize; fj++)
                            {
                                int startRow = i * stride + fi;
                                int startCol = j * stride + fj;
                                value += input3D[startRow, startCol, c] * filters[f, fi, fj, c];
                            }
                        }
                    }

                    value += biases[f]; // Add bias
                    NeuronValues[index++] = Activate(value);
                }
            }
        }
    }

    public override void Train(float[] desiredValues, float learningRate)
    {
        int outputHeight = (inputHeight - filterSize) / stride + 1;
        int outputWidth = (inputWidth - filterSize) / stride + 1;
        Errors = new float[NeuronValues.Length];

        // Calculate errors (difference between output and desired values)
        for (int i = 0; i < Errors.Length; i++)
        {
            Errors[i] = NeuronValues[i] - desiredValues[i];
        }

        // Gradients for filters and biases
        float[,,,] filterGradient = new float[numFilters, filterSize, filterSize, inputChannels];
        float[] biasGradient = new float[numFilters];

        // Backpropagate errors to update filters and biases
        for (int f = 0; f < numFilters; f++)
        {
            for (int i = 0; i < outputHeight; i++)
            {
                for (int j = 0; j < outputWidth; j++)
                {
                    int index = f * outputHeight * outputWidth + i * outputWidth + j;
                    float error = Errors[index];

                    for (int c = 0; c < inputChannels; c++)
                    {
                        for (int fi = 0; fi < filterSize; fi++)
                        {
                            for (int fj = 0; fj < filterSize; fj++)
                            {
                                int startRow = i * stride + fi;
                                int startCol = j * stride + fj;
                                filterGradient[f, fi, fj, c] += error * PreviousLayer.NeuronValues[startRow * inputWidth + startCol];
                            }
                        }
                    }

                    biasGradient[f] += error;
                }
            }
        }

        // Update filters and biases using gradients
        for (int f = 0; f < numFilters; f++)
        {
            for (int c = 0; c < inputChannels; c++)
            {
                for (int i = 0; i < filterSize; i++)
                {
                    for (int j = 0; j < filterSize; j++)
                    {
                        filters[f, i, j, c] -= learningRate * filterGradient[f, i, j, c];
                    }
                }
            }
            biases[f] -= learningRate * biasGradient[f];
        }
    }

    public override void Summary()
    {
        Console.WriteLine($"Convolutional Layer: {numFilters} filters of size {filterSize}x{filterSize} with stride {stride}");
    }

    public override void Save(BinaryWriter bw)
    {
        // Implement save logic for filters and biases
    }

    public override void Load(BinaryReader br)
    {
        // Implement load logic for filters and biases
    }

    public override void InitializeCuda(int index)
    {
        // CUDA initialization not implemented in this sample.
    }

    private float Activate(float value)
    {
        return value > 0 ? value : 0; // ReLU Activation
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