﻿
using NNFromScratch.Helper;
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
    public int filterHeight = 3;
    public int filterWidth = 3;

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
        if (this.PreviousLayer is InputLayer inputLayer)
        {
            ApplyFilters(inputLayer.NeuronValues, imageWidth, imageHeight, filters);
        }
    }
    public override void Initialize()
    {
        int outputRows = imageHeight - filterHeight + 1;
        int outputCols = imageWidth - filterWidth + 1;
        featureMap = new float[(outputRows * outputCols * 3) * filters.Length];

    }

    public override void InitializeCuda(int index)
    {
        throw new NotImplementedException();
    }
    public override void Load(BinaryReader br)
    {
        //initialize only here, the normal initialisation will be made through a function
        errorGradients = new float[featureMap.Length];

        LayerSaveLoadHelper.LoadData(featureMap, br);
        LayerSaveLoadHelper.LoadData(errorGradients, br);
    }
    public override void Save(BinaryWriter bw)
    {
        LayerSaveLoadHelper.SaveData(featureMap, bw);
        LayerSaveLoadHelper.SaveData(errorGradients, bw);
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
        int filterSize = filterGradient.Length;

        Parallel.For(0, filterSize, (idx) =>
        {
            for (int j = 0; j < featureMap.Length; j++)
            {
                featureMap[j] -= learningRate * filterGradient[idx];
            }
        });
    }

    private float[] ExtractInputSection(float[] inputImage, int inputWidth, int inputHeight,
                                     int outputX, int outputY, int stride)
    {
        float[] inputSection = new float[filterWidth * filterHeight];

        int inputX = outputX * stride;
        int inputY = outputY * stride;

        for (int i = 0; i < filterWidth; i++)
        {
            for (int y = 0; y < filterHeight; y++)
            {
                if (inputX + i < inputWidth && inputY + y < inputHeight)
                {
                    inputSection[i * filterWidth + y] = inputImage[(inputY + y) * inputWidth + (inputX + i)];
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

        Parallel.For(0, outputHeight, (idx) =>
        {
            for (int j = 0; j < outputWidth; j++)
            {
                // Extract the input section corresponding to the current output pixel
                float[] inputSection = ExtractInputSection(
                    this.PreviousLayer.NeuronValues,
                    imageWidth,
                    imageHeight,
                    j,
                    idx,
                    stride);

                for (int k = 0; k < filterSize; k++)
                {
                    filterGradient[k] += errorGradient[idx * outputWidth + j] * inputSection[k];
                }
            }
        });

        return filterGradient;
    }
    private float[] CalculateOutputGradients(float[] desiredValues, float[] pooledOutput)
    {
        // Create a gradient array for the output layer
        float[] outputGradients = new float[desiredValues.Length];

        // Calculate the output layer's gradients based on desired values and actual pooled output
        for (int i = 0; i < desiredValues.Length; i++)
        {
            outputGradients[i] = desiredValues[i] - pooledOutput[i];
        }

        // Now distribute the gradients back to the full feature map
        float[] fullGradients = new float[featureMap.Length];

        int outputWidth = featureMapX; // The width of your feature map
        int outputHeight = featureMapY; // The height of your feature map

        // Assuming each output gradient influences a region in the feature map, distribute accordingly
        int regionWidth = outputWidth / desiredValues.Length; // Divide feature map among outputs
        int regionHeight = outputHeight / desiredValues.Length;

        Parallel.For(0, desiredValues.Length, (idx) =>
        {
            for (int y = 0; y < regionHeight; y++)
            {
                for (int x = 0; x < regionWidth; x++)
                {
                    // Mapping the output gradient to the corresponding region of the feature map
                    int featureMapIndex = (idx * regionWidth * regionHeight) + (y * regionWidth + x);
                    fullGradients[featureMapIndex] += outputGradients[idx]; // Distribute the gradient
                }
            }
        });

        return fullGradients;
    }
    private float ElementWiseMultiplyRGB(float[] imageSection, float[] filter, int sectionWidth, int sectionHeight, int channel)
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
    private float[] ConvolutionRGB(float[] image, int imageWidth, int imageHeight, float[] filter)
    {
        int outputRows = imageHeight - filterHeight + 1;
        int outputCols = imageWidth - filterWidth + 1;
        float[] output = new float[outputRows * outputCols * 3]; //r,g,b

        //iterate over every pixel of the image and apply filter
        Parallel.For(0, outputRows, idx =>
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
                                image[((idx + x) * imageWidth + (j + y)) * 3 + channel];
                        }
                    }
                }

                for (int channel = 0; channel < 3; channel++)
                {
                    output[((idx * outputCols) + j) * 3 + channel] = ElementWiseMultiplyRGB(imageSection, filter, filterWidth, filterHeight, channel);
                }
            }
        });

        return output;
    }
    private void ApplyFilters(float[] image, int imageWidth, int imageHeight, ConvolutionalFilterType[] filterType)
    {
        for (int i = 0; i < filters.Length; i++)
        {
            var extractedFeatures = ConvolutionRGB(image, imageWidth, imageHeight, GetFilterForType(filters[i]));

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
            case ConvolutionalFilterType.Embossing:
                return new float[] { -2, -1, 0, -1, 1, 1, 0, 1, 2 };  //3x3 Embossing
            case ConvolutionalFilterType.Sharpening:
                return new float[] { 0, -1, 0, -1, 5, -1, 0, -1, 0 };  //3x3 Sharpening
            default:
                throw new ArgumentException("Unknown filter in Convolutional layer");
        }
    }
}