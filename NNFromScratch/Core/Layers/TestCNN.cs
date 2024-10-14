namespace NNFromScratch.Core.Layers;

internal class TestCNN
{
    public static int ElementWiseMultiply(int[,] imageSection, int[,] filter)
    {
        int sum = 0;
        int rows = imageSection.GetLength(0);
        int cols = imageSection.GetLength(1);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                sum += imageSection[i, j] * filter[i, j];
            }
        }

        return sum;
    }
    public static int[,] Convolution(int[,] image, int[,] filter)
    {
        int outputRows = image.GetLength(0) - filter.GetLength(0) + 1;
        int outputCols = image.GetLength(1) - filter.GetLength(1) + 1;
        int[,] output = new int[outputRows, outputCols];

        for (int i = 0; i < outputRows; i++)
        {
            for (int j = 0; j < outputCols; j++)
            {
                // Extract 3x3 section of the image
                int[,] imageSection = new int[filter.GetLength(0), filter.GetLength(1)];
                for (int x = 0; x < filter.GetLength(0); x++)
                {
                    for (int y = 0; y < filter.GetLength(1); y++)
                    {
                        imageSection[x, y] = image[i + x, j + y];
                    }
                }

                // Perform convolution using the element-wise multiplication function
                output[i, j] = ElementWiseMultiply(imageSection, filter);
            }
        }

        return output;
    }

    public static void Main()
    {
        int[,] image = {
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}
        };

        int[,] filter = {
            {1, 0, -1},
            {1, 0, -1},
            {1, 0, -1}
        };  

        int[,] result = Convolution(image, filter);

        // Output the result
        for (int i = 0; i < result.GetLength(0); i++)
        {
            for (int j = 0; j < result.GetLength(1); j++)
            {
                Console.Write(result[i, j] + " ");
            }
            Console.WriteLine();
        }
    }
}
