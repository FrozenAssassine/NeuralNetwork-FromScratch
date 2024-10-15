
using NNFromScratch.Models;

namespace NNFromScratch.Core.Layers;

public class ConvolutionalLayer : BaseLayer
{
    public float[] featureMap;
    public int imageWidth = 0;
    public int imageHeight = 0;
    public float[] errorGradients;
    public ConvolutionalFilterType[] filters;
    public int featureMapX;
    public int featureMapY;
    public int stride;

    public ConvolutionalLayer(int imageWidth, int imageHeight, int stride, int featureMapX, int featureMapY, ConvolutionalFilterType[] filters)
    {
        this.imageWidth = imageWidth;
        this.imageHeight = imageHeight;
        this.filters = filters;
        this.stride = stride;
        this.featureMapX = featureMapX;
        this.featureMapY = featureMapY;
    }

    public override void FeedForward()
    {
        if(this.PreviousLayer is InputLayer inputLayer)
        {
            ApplyFilters(inputLayer.NeuronValues, imageWidth, imageHeight, filters);
        }
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
        errorGradients = CalculateOutputGradients(desiredValues, featureMap);

        //update the gradients for every filter:
        for (int i = 0; i < filters.Length; i++)
        {
            float[] gradients = CalculateFilterGradient(errorGradients, i);
            UpdateFilter(gradients, learningRate);
        }
    }
    private void UpdateFilter(float[] filterGradient, float learningRate)
    {
        // Assuming featureMap is organized by filter and the gradient affects specific regions.
        int outputWidth = featureMapX; // Width of the feature map
        int outputHeight = featureMapY; // Height of the feature map
        int filterSize = filterGradient.Length; // Should be the number of output classes (6 in your case)

        // Loop through the output neurons (which is 6)
        for (int i = 0; i < filterSize; i++)
        {
            // Determine the position in the featureMap to update
            int outputX = i % outputWidth; // Assuming linear access
            int outputY = i / outputWidth;  // Assuming linear access

            // Propagate the filterGradient back to the relevant regions in featureMap
            for (int j = 0; j < featureMap.Length; j++)
            {
                // Only update the relevant sections of the featureMap
                // Example logic, replace with your actual logic to determine the effect region
                // This assumes each filter affects a specific portion of the feature map based on its position
                featureMap[j] -= learningRate * filterGradient[i]; // Adjust this logic as per your filter mapping
            }
        }
    }

    private float[] ExtractInputSection(float[] inputImage, int inputWidth, int inputHeight,
                                     int filterWidth, int filterHeight,
                                     int outputX, int outputY, int stride)
    {
        float[] inputSection = new float[filterWidth * filterHeight];

        int inputX = outputX * stride;
        int inputY = outputY * stride;

        for (int x = 0; x < filterWidth; x++)
        {
            for (int y = 0; y < filterHeight; y++)
            {
                if (inputX + x < inputWidth && inputY + y < inputHeight)
                {
                    inputSection[x * filterWidth + y] = inputImage[(inputY + y) * inputWidth + (inputX + x)];
                }
            }
        }

        return inputSection;
    }
    private float[] CalculateFilterGradient(float[] errorGradient, int filterIndex)
    {
        float[] filter = GetFilterForType(filters[filterIndex]);

        int filterSize = filter.Length;
        int outputWidth = featureMapX;
        int outputHeight = featureMapY;

        float[] filterGradient = new float[filterSize];

        for (int i = 0; i < outputHeight; i++)
        {
            for (int j = 0; j < outputWidth; j++)
            {
                // Extract the input section corresponding to the current output pixel
                float[] inputSection = ExtractInputSection(
                    this.PreviousLayer.NeuronValues,
                    imageWidth,
                    imageHeight,
                    3,
                    3,
                    j,
                    i,
                    stride);

                for (int k = 0; k < filterSize; k++)
                {
                    filterGradient[k] += errorGradient[i * outputWidth + j] * inputSection[k];
                }
            }
        }

        return filterGradient;
    }
    private float[] CalculateOutputGradients(float[] desiredValues, float[] pooledOutput)
    {
        float[] outputGradients = new float[desiredValues.Length];

        for (int i = 0; i < desiredValues.Length; i++)
        {
            outputGradients[i] = desiredValues[i] - pooledOutput[i];
        }

        // Create a larger gradient array for the feature map
        float[] fullGradients = new float[featureMap.Length];

        // Distribute output gradients to the full gradients
        for (int i = 0; i < desiredValues.Length; i++)
        {
            int outputIndex = i; // You need to map your output neuron to the corresponding feature map region

            // Adjust based on your feature map size and how the outputs correspond to it
            for (int j = 0; j < featureMapY; j++)
            {
                for (int k = 0; k < featureMapX; k++)
                {
                    var mappingFactor = 1f / (featureMapX * featureMapY); // Normalizes the contribution

                    // You would have to determine how this mapping works, for example:
                    fullGradients[outputIndex] += outputGradients[i] * mappingFactor;
                }
            }
        }

        return fullGradients;
    }
    public static float ElementWiseMultiplyRGB(float[] imageSection, float[] filter, int sectionWidth, int sectionHeight, int channel)
    {
        float sum = 0;
        for (int i = 0; i < sectionHeight; i++)
        {
            for (int j = 0; j < sectionWidth; j++)
            {
                int index = (i * sectionWidth + j) * 3 + channel;
                sum += imageSection[index] * filter[i * sectionWidth + j];
            }
        }
        return sum;
    }
    public static float[] ConvolutionRGB(float[] image, int imageWidth, int imageHeight, float[] filter, int filterWidth, int filterHeight)
    {
        int outputRows = imageHeight - filterHeight + 1;
        int outputCols = imageWidth - filterWidth + 1;
        float[] output = new float[outputRows * outputCols * 3]; //r,g,b

        //iterate over every pixel of the image and apply filter
        for (int i = 0; i < outputRows; i++)
        {
            for (int j = 0; j < outputCols; j++)
            {
                float[] imageSection = new float[filterWidth * filterHeight * 3];
                for (int x = 0; x < filterHeight; x++)
                {
                    for (int y = 0; y < filterWidth; y++)
                    {
                        for (int channel = 0; channel < 3; channel++)
                        {
                            imageSection[(x * filterWidth + y) * 3 + channel] =
                                image[((i + x) * imageWidth + (j + y)) * 3 + channel];
                        }
                    }
                }

                for (int channel = 0; channel < 3; channel++)
                {
                    output[((i * outputCols) + j) * 3 + channel] = ElementWiseMultiplyRGB(imageSection, filter, filterWidth, filterHeight, channel);
                }
            }
        }

        return output;
    }
    public void ApplyFilters(float[] image, int imageWidth, int imageHeight, ConvolutionalFilterType[] filterType)
    {
        for (int i = 0; i < filters.Length; i++)
        {
            var extractedFeatures = ConvolutionRGB(image, imageWidth, imageHeight, GetFilterForType(filters[i]), 3, 3);
            if (featureMap == null)
                featureMap = new float[extractedFeatures.Length * filters.Length];

            int offset = extractedFeatures.Length * i; //check whether correct?
            Array.Copy(extractedFeatures, 0, featureMap, offset, extractedFeatures.Length);
        }
    }
    private float[] GetFilterForType(ConvolutionalFilterType filterType)
    {
        switch (filterType)
        {
            case ConvolutionalFilterType.SobelX:
                return new float[] { -1, 0, 1, -2, 0, 2, -1, 0, 1 }; //3x3 Sobel X
            case ConvolutionalFilterType.SobelY:
                return new float[] { -1, -2, -1, 0, 0, 0, 1, 2, 1 }; //3x3 Sobel Y
            case ConvolutionalFilterType.Laplacian:
                return new float[] { 0, 1, 0, 1, -4, 1, 0, 1, 0 };  //3x3 Laplacian
            default:
                throw new ArgumentException("Unknown filter in Convolutional layer");
        }
    }
}
