
namespace NNFromScratch.Core.Layers;

public class ConvolutionalLayer : BaseLayer
{
    private float[] filters;
    private float[] biases;

    public int NumFilters { get; }
    public int FilterSize { get; }
    public int InputHeight { get; }
    public int InputWidth { get; }
    public int OutputHeight { get; }
    public int OutputWidth { get; }

    public ConvolutionalLayer(int inputHeight, int inputWidth, int filterSize, int numFilters)
    {
        NumFilters = numFilters;
        FilterSize = filterSize;
        InputHeight = inputHeight;
        InputWidth = inputWidth;

        // Calculate output dimensions
        OutputHeight = inputHeight - filterSize + 1;
        OutputWidth = inputWidth - filterSize + 1;

        // Initialize filters and biases
        filters = InitializeRandomArray(numFilters, filterSize, filterSize);
        biases = InitializeRandomArray(numFilters, OutputHeight, OutputWidth);
    }

    private float[] InitializeRandomArray(int depth, int height, int width)
    {
        var array = new float[depth * height * width];
        var random = new Random();

        for (int d = 0; d < depth; d++)
        {
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    array[d  * height * width + h * width + w] = random.NextSingle();
                }
            }
        }

        return array;
    }

    public override void FeedForward()
    {
        var output = new float[OutputWidth * OutputHeight];

        // Loop through each filter
        for (int i = 0; i < NumFilters; i++)
        {
            // Get the current filter (assuming you have a filters array)

            // Calculate the correlation output for the current filter
            float[] filterOutput = Correlate2D(this.PreviousLayer.NeuronValues, InputHeight, filters, FilterSize, FilterSize);

            // Assign the output for this filter
            Array.Copy(filterOutput, 0, output, i * OutputWidth * OutputHeight, filterOutput.Length);
        }

        // Store the output for the next layer (not shown in your code)
        this.NeuronValues = output;
    }

    private float[] Correlate2D(float[] input, int imageHeight, float[] filter, int filterWidth, int filterHeight)
    {
        int inputHeight = imageHeight;
        int inputWidth = input.Length / imageHeight;

        int outputHeight = inputHeight - filterHeight + 1;
        int outputWidth = inputWidth - filterWidth + 1;

        float[] output = new float[outputWidth * outputHeight];

        // Perform 2D correlation (sliding window)
        for (int i = 0; i < outputHeight; i++)
        {
            for (int j = 0; j < outputWidth; j++)
            {
                float sum = 0.0f;

                // Element-wise multiplication of the filter and the input patch
                for (int m = 0; m < filterHeight; m++)
                {
                    for (int n = 0; n < filterWidth; n++)
                    {
                        // Calculate the 1D index for the input array
                        int inputIndex = (i + m) * inputWidth + (j + n);
                        int filterIndex = m * filterWidth + n;

                        sum += input[inputIndex] * filter[filterIndex];
                    }
                }

                output[i * outputWidth + j] = sum;
            }
        }

        return output;
    }

    public override void Initialize()
    {

    }

    public override void InitializeCuda(int index)
    {
        throw new NotImplementedException();
    }

    public override void Load(BinaryReader br)
    {
        throw new NotImplementedException();
    }

    public override void Save(BinaryWriter bw)
    {
        throw new NotImplementedException();
    }

    public override void Summary()
    {
        throw new NotImplementedException();
    }

    public override void Train(float[] desiredValues, float learningRate)
    {
        throw new NotImplementedException();
    }
}