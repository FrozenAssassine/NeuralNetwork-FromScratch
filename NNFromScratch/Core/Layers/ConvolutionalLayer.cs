
using NNFromScratch.Helper;

namespace NNFromScratch.Core.Layers;

public class ConvolutionalLayer : BaseLayer
{
    private readonly int numberFilters;
    private readonly int kernelSize;
    private readonly int padding;
    private readonly int strideW;
    private readonly int strideH;
    private (int imageWidth, int imageHeight) inputSize;
    private (int width, int height) outputSize;

    public ConvolutionalLayer(int numberFilters, (int imageWidth, int imageHeight) inputSize, int kernelSize, int padding, ActivationType activation = ActivationType.Relu, int strideW = 1, int strideH = 1)
    {
        this.numberFilters = numberFilters;
        this.kernelSize = kernelSize;
        this.padding = padding;
        this.strideH = strideH;
        this.strideW = strideW;
        this.inputSize = inputSize;
    }

    private (int height, int width) GetOutputShape2((int imageWidth, int imageHeight) inputSize, (int width, int height) filterShape, int padding, int strideW, int strideH)
    {
        return (height: (inputSize.imageHeight - filterShape.height + padding * 2 + strideH) / strideH,
            width: (inputSize.imageWidth - filterShape.width + padding * 2 + strideW) / strideW);
    }

    private (int height, int width) GetOutputShape((int imageWidth, int imageHeight) inputSize, (int width, int height) filterShape, int padding, int strideW, int strideH)
    {
        return (height: (inputSize.imageHeight - filterShape.height + 2 * padding) / strideH + 1,
            width: (inputSize.imageWidth - filterShape.width + 2 * padding) / strideW + 1);
    }

    private float[] ResizePicture(float[] input, int padding)
    {
        var paddedWidth = this.inputSize.imageWidth + padding * 2;
        var paddedHeight = this.inputSize.imageHeight + padding * 2;
        float[] output = new float[paddedWidth * paddedHeight];

        // Initialize with zeros (for padding)
        for (int i = 0; i < output.Length; i++)
        {
            output[i] = 0.0f;
        }

        // Copy the input values to the padded output
        for (int y = 0; y < this.inputSize.imageHeight; y++)
        {
            for (int x = 0; x < this.inputSize.imageWidth; x++)
            {
                int inputIndex = y * this.inputSize.imageWidth + x;
                int outputIndex = (y + padding) * paddedWidth + (x + padding);
                output[outputIndex] = input[inputIndex];
            }
        }
        return output;
    }


    private List<(float[], int, int)> GetPortions(float[] input)
    {
        var paddedWidth = this.inputSize.imageWidth + padding * 2;
        var resizedPicture = ResizePicture(input, padding);
        var result = new List<(float[], int, int)>();

        for (int y = 0; y < outputSize.height; y++)
        {
            for (int x = 0; x < outputSize.width; x++)
            {
                int startX = x * strideW;
                int startY = y * strideH;
                var portion = GetPortion(resizedPicture, paddedWidth, startX, startY);
                result.Add((portion, x, y));
            }
        }

        return result;
    }

    private float[] GetPortion(float[] input, int paddedWidth, int startX, int startY)
    {
        float[] portion = new float[kernelSize * kernelSize];

        for (int y = 0; y < kernelSize; y++)
        {
            for (int x = 0; x < kernelSize; x++)
            {
                int inputIndex = (startY + y) * paddedWidth + (startX + x);
                int outputIndex = y * kernelSize + x;
                portion[outputIndex] = input[inputIndex];
            }
        }

        return portion;
    }

    private float ConvolveOperation(float[] input, float[] filter, float bias)
    {
        float sum = 0;

        for (int i = 0; i < input.Length; i++)
        {
            sum += input[i] * filter[i];
        }

        return sum + bias;
    }


    public override void FeedForward()
    {
        var portions = GetPortions(PreviousLayer.NeuronValues);
        
        int outSize = this.outputSize.width * this.outputSize.height * numberFilters;
        NeuronValues = new float[outSize];

        // For each filter
        for (int f = 0; f < numberFilters; f++)
        {
            // Get the filter weights
            float[] filter = new float[kernelSize * kernelSize];
            int filterOffset = f * kernelSize * kernelSize;

            for (int i = 0; i < filter.Length; i++)
            {
                filter[i] = Weights[filterOffset + i];
            }

            float bias = Biases[f];

            // Apply filter to each portion
            foreach (var (portion, x, y) in portions)
            {
                float value = ConvolveOperation(portion, filter, bias);
                float activatedValue = ActivationFunctions.Activation(value, this.ActivationFunction);

                // Store the result in the output array
                int outputIndex = f * (outputSize.width * outputSize.height) + y * outputSize.width + x;
                NeuronValues[outputIndex] = activatedValue;
            }
        }
    }


    private float[] GetFilter(int filterIndex)
    {
        float[] filter = new float[kernelSize * kernelSize];
        int filterOffset = filterIndex * kernelSize * kernelSize;

        for (int i = 0; i < filter.Length; i++)
        {
            filter[i] = Weights[filterOffset + i];
        }

        return filter;
    }

    private void UpdateWeightsAndBiases(List<(float[], int, int)> portions, float learningRate)
    {
        // For each filter
        for (int f = 0; f < numberFilters; f++)
        {
            float biasDelta = 0;
            float[] filterGradients = new float[kernelSize * kernelSize];

            // Calculate gradients across all portions
            foreach (var (portion, x, y) in portions)
            {
                int outputIndex = f * (outputSize.width * outputSize.height) + y * outputSize.width + x;
                float error = Errors[outputIndex];

                // Update bias gradient
                biasDelta += error;

                // Update filter gradients
                for (int i = 0; i < portion.Length; i++)
                {
                    filterGradients[i] += portion[i] * error;
                }
            }

            // Apply updates
            Biases[f] -= learningRate * biasDelta;

            int filterOffset = f * kernelSize * kernelSize;
            for (int i = 0; i < filterGradients.Length; i++)
            {
                Weights[filterOffset + i] -= learningRate * filterGradients[i];
            }
        }
    }

    private void PropagateErrorsToPreviousLayer()
    {
        if (PreviousLayer == null)
            return;

        // Reset previous layer errors
        Array.Clear(PreviousLayer.Errors, 0, PreviousLayer.Errors.Length);

        var paddedWidth = this.inputSize.imageWidth + padding * 2;

        // For each filter
        for (int f = 0; f < numberFilters; f++)
        {
            float[] filter = GetFilter(f);

            // For each position in the output
            for (int y = 0; y < outputSize.height; y++)
            {
                for (int x = 0; x < outputSize.width; x++)
                {
                    int outputIndex = f * (outputSize.width * outputSize.height) + y * outputSize.width + x;
                    float error = Errors[outputIndex];

                    PropagateErrorForPosition(error, filter, x, y);
                }
            }
        }
    }

    private void PropagateErrorForPosition(float error, float[] filter, int x, int y)
    {
        for (int ky = 0; ky < kernelSize; ky++)
        {
            for (int kx = 0; kx < kernelSize; kx++)
            {
                int inputY = y * strideH + ky - padding;
                int inputX = x * strideW + kx - padding;

                // Check if the input position is valid (not in padding)
                if (inputY >= 0 && inputY < inputSize.imageHeight &&
                    inputX >= 0 && inputX < inputSize.imageWidth)
                {
                    int inputIndex = inputY * inputSize.imageWidth + inputX;
                    int filterIndex = ky * kernelSize + kx;

                    PreviousLayer.Errors[inputIndex] += error * filter[filterIndex];
                }
            }
        }
    }

    private void CalculateOutputErrors(float[] desiredValues)
    {
        for (int i = 0; i < Size; i++)
        {
            float error = desiredValues[i] - NeuronValues[i];
            Errors[i] = error * ActivationFunctions.ActivationDeriv(NeuronValues[i], ActivationFunction);
        }
    }

    public override void Train(float[] desiredValues, float learningRate)
    {
        // Calculate output errors if this is the output layer
        if (NextLayer == null)
        {
            CalculateOutputErrors(desiredValues);
        }

        // Get portions from the previous layer for weight updates
        var portions = GetPortions(PreviousLayer.NeuronValues);

        // Update weights and biases based on errors
        UpdateWeightsAndBiases(portions, learningRate);

        // Propagate errors to previous layer
        PropagateErrorsToPreviousLayer();
    }

    public override void Initialize(int inputCount, int outputCount)
    {
        if (!(this.PreviousLayer is InputLayer || this.PreviousLayer is MaxPoolingLayer))
        {
            throw new Exception("Convolutional Layer either needs InputLayer or MaxPoolingLayer in front of it!");
        }

        this.outputSize = GetOutputShape(inputSize, (width: kernelSize, height: kernelSize), padding, strideW, strideH);

        // Calculate the size of this layer's output
        Size = outputSize.width * outputSize.height * numberFilters;

        // Initialize arrays
        Biases = new float[numberFilters];
        NeuronValues = new float[Size];
        Errors = new float[Size];

        // Each filter has kernelSize x kernelSize weights
        Weights = new float[numberFilters * kernelSize * kernelSize];

        LayerInitialisationHelper.InitializeLayer(this, inputCount, outputCount, numberFilters * kernelSize * kernelSize);

        // Initialize biases to zero
        for (int i = 0; i < Biases.Length; i++)
        {
            Biases[i] = 0;
        }
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
        Console.WriteLine($"Convolutional Layer {inputSize}in => {outputSize}out => {Weights.Length}weights");
    }

}
