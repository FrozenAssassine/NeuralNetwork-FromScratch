namespace NNFromScratch.Core.Layers;

public class TestCNN
{
    //convolutional layer:
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
    public static float ElementWiseMultiplyBW(float[] imageSection, int imageWidth, int imageHeight, float[] filter)
    {
        float sum = 0;
        for (int i = 0; i < imageWidth; i++)
        {
            for (int j = 0; j < imageHeight; j++)
            {
                sum += imageSection[(i * imageWidth) + j] * filter[(i * imageWidth) + j];
            }
        }

        return sum;
    }
    public static float[] ConvolutionRGB(float[] image, int imageWidth, int imageHeight, float[] filter, int filterWidth, int filterHeight)
    {
        int outputRows = imageHeight - filterHeight + 1;
        int outputCols = imageWidth - filterWidth + 1;
        float[] output = new float[outputRows * outputCols * 3]; //r,g,b

        //iterate over every pixel of the image
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

    //pooling layer: 
    public static float[] Forward(float[] input, int inputWidth, int inputHeight, int poolSize, int stride)
    {
        int outputWidth = (inputWidth - poolSize) / stride + 1;
        int outputHeight = (inputHeight - poolSize) / stride + 1;
        float[] output = new float[outputWidth * outputHeight];

        for (int y = 0; y < outputHeight; y++)
        {
            for (int x = 0; x < outputWidth; x++)
            {
                float sum = 0;
                int count = 0;

                // Apply pooling
                for (int j = 0; j < poolSize; j++)
                {
                    for (int i = 0; i < poolSize; i++)
                    {
                        int inputX = x * stride + i;
                        int inputY = y * stride + j;

                        // Ensure we're within the bounds of the input
                        if (inputX < inputWidth && inputY < inputHeight)
                        {
                            sum += input[inputY * inputWidth + inputX];
                            count++;
                        }
                    }
                }

                // Calculate the average
                output[y * outputWidth + x] = sum / count;
            }
        }

        return output;
    }

    public static void Main()
    {
        float[] image = {
            255, 0, 0,   0, 255, 0,   0, 255, 0,
            255, 255, 0,   0, 255, 255,   0, 255, 0,
            0, 0, 255,   255, 255, 255,   0, 255, 0,
        };

        //edge detection filter (horizontal)
        float[] filter = {
            -1, 0, 1,
            -2, 0, 2,
            -1, 0, 1
        };

        float[] res = ConvolutionRGB(image, 3,3, filter, 3, 3);
        for (int i = 0; i < res.Length; i += 3)
        {
            Console.WriteLine($"R: {res[i]}, G: {res[i + 1]}, B: {res[i + 2]}");
        }


        //pooling:
        float[] input = new float[]
        {
            1, 2, 3, 0,
            5, 6, 7, 8,
            9, 10, 11, 12,
            13, 14, 15, 16
        };
        
        int inputWidth = 4;
        int inputHeight = 4;
        float[] pooledOutput = Forward(input, inputWidth, inputHeight, 2, 2);
        
    }
}
