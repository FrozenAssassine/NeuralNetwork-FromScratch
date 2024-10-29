using NNFromScratch.Helper;
using NNFromScratch.Models;
using System.Diagnostics;
using System.Drawing.Imaging;
using System.Drawing;

namespace NNFromScratch.Core.Layers;

public class ConvolutionalLayer : BaseLayer
{
    public float[] featureMap;
    public int imageWidth;
    public int imageHeight;
    public float[] errorGradients;
    public ConvolutionalFilterType[] filters;
    public int featureMapX;
    public int featureMapY;
    public int stride;
    public int filterHeight = 3;
    public int filterWidth = 3;
    public int padding = 0;

    public ConvolutionalLayer(int imageWidth, int imageHeight, int stride, ConvolutionalFilterType[] filters)
    {
        this.imageWidth = imageWidth;
        this.imageHeight = imageHeight;
        this.filters = filters;
        this.stride = stride;

        var fm = CalculateFeatureMapSize();
        this.featureMapX = fm.width;
        this.featureMapY = fm.height;
    }

    public (int width, int height) CalculateFeatureMapSize()
    {
        int featureMapWidth = (imageWidth - filterWidth + 2 * padding) / stride + 1;
        int featureMapHeight = (imageHeight - filterHeight + 2 * padding) / stride + 1;
        return (featureMapWidth, featureMapHeight);
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
        featureMap = new float[featureMapX * featureMapY * 3 * filters.Length];
    }

    public override void InitializeCuda(int index)
    {
        throw new NotImplementedException();
    }

    public override void Load(BinaryReader br)
    {
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
        Console.WriteLine($"Convolutional Layer of {this.featureMap.Length} features.");
        Console.WriteLine($"\tFeatureMap: ({featureMapX}|{featureMapY})");
        Console.WriteLine($"\tImage: ({imageWidth}|{imageHeight})");
        Console.WriteLine($"\tFilter: {string.Join(", ", filters)}");
    }

    public override void Train(float[] desiredValues, float learningRate)
    {
        errorGradients = CalculateOutputGradients(NextLayer.NeuronValues, featureMap, 2);

        for (int i = 0; i < filters.Length; i++)
        {
            float[] gradients = CalculateFilterGradient(errorGradients, i);
            UpdateFilter(gradients, learningRate);
        }
    }

    private void UpdateFilter(float[] filterGradient, float learningRate)
    {
        int filterSize = filterGradient.Length;
        Parallel.For(0, filterSize, idx =>
        {
            for (int j = 0; j < featureMap.Length; j++)
            {
                featureMap[j] -= learningRate * filterGradient[idx];
            }
        });
    }

    private float[] CalculateFilterGradient(float[] errorGradient, int filterIndex)
    {
        float[] filter = GetFilterForType(filters[filterIndex]);
        int filterSize = filter.Length;
        int outputWidth = featureMapX;
        int outputHeight = featureMapY;
        float[] filterGradient = new float[filterSize];

        Parallel.For(0, outputHeight, idx =>
        {
            for (int j = 0; j < outputWidth; j++)
            {
                float[] inputSection = ExtractInputSection(this.PreviousLayer.NeuronValues, imageWidth, imageHeight, j, idx);
                for (int k = 0; k < filterSize; k++)
                {
                    filterGradient[k] += errorGradient[idx * outputWidth + j] * inputSection[k];
                }
            }
        });
        return filterGradient;
    }

    private float[] CalculateOutputGradients(float[] desiredValues, float[] featureMap)
    {
        float[] outputGradients = new float[desiredValues.Length];

        for (int i = 0; i < desiredValues.Length; i++)
        {
            outputGradients[i] = desiredValues[i] - featureMap[i];
        }

        float[] fullGradients = new float[this.featureMap.Length];
        int regionWidth = featureMapX / desiredValues.Length;
        int regionHeight = featureMapY / desiredValues.Length;

        Parallel.For(0, desiredValues.Length, idx =>
        {
            for (int y = 0; y < regionHeight; y++)
            {
                for (int x = 0; x < regionWidth; x++)
                {
                    int featureMapIndex = (idx * regionWidth * regionHeight) + (y * regionWidth + x);
                    fullGradients[featureMapIndex] += outputGradients[idx];
                }
            }
        });
        return fullGradients;
    }
    private float[] ExtractInputSection(float[] inputImage, int inputWidth, int inputHeight, int outputX, int outputY)
    {
        //extract 3x3 sections from the image:
        float[] inputSection = new float[filterWidth * filterHeight * 3];
        int inputX = outputX * stride;
        int inputY = outputY * stride;

        for (int y = 0; y < filterHeight; y++)
        {
            for (int x = 0; x < filterWidth; x++)
            {
                for (int channel = 0; channel < 3; channel++) //rgb
                {
                    int sectionIndex = (y * filterWidth + x) * 3 + channel;
                    int imageIndex = ((inputY + y) * inputWidth + (inputX + x)) * 3 + channel;

                    //if ((inputX + x) < inputWidth && (inputY + y) < inputHeight)
                        inputSection[sectionIndex] = inputImage[imageIndex];
                    //else
                        //inputSection[sectionIndex] = -1; 
                }
            }
        }
        return inputSection;
    }


    //add together all image image pixels of the filter size section (e.g. 3x3xrgb = 27) and apply the filter:
    private float ElementWiseMultiplyRGB(float[] imageSection, float[] filter, int channel)
    {
        float sum = 0;
        for (int i = 0; i < filterHeight; i++)
        {
            for (int j = 0; j < filterWidth; j++)
            {
                int index = (i * filterWidth + j) * 3 + channel;
                sum += imageSection[index] * filter[i * filterWidth + j];
            }
        }
        return sum;
    }
    private void ApplyFilters(float[] image, int imageWidth, int imageHeight, ConvolutionalFilterType[] filterType)
    {
        for (int i = 0; i < filters.Length; i++)
        {
            ConvolutionRGB(GetFilterForType(filters[i]), image, i);
            //FeatureMapToImage.SaveImage(featureMap, i * (featureMapX * featureMapY * 3), featureMapX, featureMapY, $"C:\\Users\\juliu\\Desktop\\image{i}.png");
        }
    }
    private void ConvolutionRGB(float[] filter, float[] image, int filterIndex)
    {
        var (outputCols, outputRows) = CalculateFeatureMapSize();
        int lastIDX = -1;
        for (int row = 0; row < outputRows; row++)
        {
            for (int col = 0; col < outputCols; col++)
            {
                for (int channel = 0; channel < 3; channel++)  // Iterate over R, G, B channels
                {
                    // Calculate the output index in the featureMap for the current position and channel
                    int outputIdx = ((row * outputCols) + col) * 3 + channel
                                    + filterIndex * outputCols * outputRows * 3;

                    var inputSection = ExtractInputSection(image, imageWidth, imageHeight, col, row);

                    // Perform element-wise multiplication and store the result directly in featureMap
                    featureMap[outputIdx] = ElementWiseMultiplyRGB(
                        inputSection,
                        filter,
                        channel
                    );
                }
            }
        }
    }
    private float[] GetFilterForType(ConvolutionalFilterType filterType)
    {
        switch (filterType)
        {
            case ConvolutionalFilterType.SobelX:
                return new float[] { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
            case ConvolutionalFilterType.SobelY:
                return new float[] { -1, -2, -1, 0, 0, 0, 1, 2, 1 };
            case ConvolutionalFilterType.Laplacian:
                return new float[] { 0, 1, 0, 1, -4, 1, 0, 1, 0 };
            case ConvolutionalFilterType.Embossing:
                return new float[] { -2, -1, 0, -1, 1, 1, 0, 1, 2 };
            case ConvolutionalFilterType.Sharpening:
                return new float[] { 0, -1, 0, -1, 5, -1, 0, -1, 0 };
            default:
                throw new ArgumentException("Unknown filter in Convolutional layer");
        }
    }
}

public class FeatureMapToImage
{
    public static Bitmap ConvertFeatureMapToRGBImage(float[] featureMap, int width, int height, int filterOffset)
    {
        Bitmap image = new Bitmap(width, height, PixelFormat.Format24bppRgb);

        // Find the min and max values in the feature map to normalize the pixel values
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int idx = filterOffset + (y * width + x) * 3;

                // Normalize each channel to 0-255
                int red = (int)(featureMap[idx] * 255);
                int green = (int)(featureMap[idx + 1] * 255);
                int blue = (int)(featureMap[idx + 2] * 255);

                // Ensure values are within bounds (0-255)
                red = Math.Clamp(red, 0, 255);
                green = Math.Clamp(green, 0, 255);
                blue = Math.Clamp(blue, 0, 255);

                // Set pixel color
                Color color = Color.FromArgb(red, green, blue);
                image.SetPixel(x, y, color);
            }
        }
        return image;
    }
    public static Bitmap PoolingToBitmap(float[] pooling, int width, int height)
    {
        Bitmap image = new Bitmap(width, height, PixelFormat.Format24bppRgb);

        // Find the min and max values in the feature map to normalize the pixel values
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int idx = (y * width + x) * 3;

                // Normalize each channel to 0-255
                int red = (int)(pooling[idx] * 255);
                int green = (int)(pooling[idx + 1] * 255);
                int blue = (int)(pooling[idx + 2] * 255);

                // Ensure values are within bounds (0-255)
                red = Math.Clamp(red, 0, 255);
                green = Math.Clamp(green, 0, 255);
                blue = Math.Clamp(blue, 0, 255);

                // Set pixel color
                Color color = Color.FromArgb(red, green, blue);
                image.SetPixel(x, y, color);
            }
        }
        return image;
    }
    public static void SaveImage(float[] featureMap, int filterOffset, int width, int height, string filePath)
    {
        using (Bitmap image = ConvertFeatureMapToRGBImage(featureMap, width, height, filterOffset))
        {
            image.Save(filePath, ImageFormat.Png);
        }
    }

    public static void SaveImagePooling(float[] pooling, int width, int height, string filePath)
    {
        using (Bitmap image = PoolingToBitmap(pooling, width, height))
        {
            image.Save(filePath, ImageFormat.Png);
        }
    }
}
