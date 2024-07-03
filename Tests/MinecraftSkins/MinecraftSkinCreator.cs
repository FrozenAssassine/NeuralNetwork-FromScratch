using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp;
using NNFromScratch.Core;
using NNFromScratch.Core.Layers;

namespace Tests.MCSkinCreator;

internal class MinecraftSkinCreator
{
    static int imageWidth = 64;
    static int imageHeight = 64;

    public static void Run()
    {
        var images = GetImages("F:\\NN DATASETS\\skins", 5000);
        float[][] x = new float[images.Length][];
        for(int i = 0;i < images.Length; i++)
        {
            x[i] = new float[] { i / 5000.0f };
        }

        var model= NetworkBuilder.Create()
            .Stack(new InputLayer(1))
            .Stack(new DenseLayer(512, ActivationType.Sigmoid))
            .Stack(new DenseLayer(512, ActivationType.Sigmoid))
            .Stack(new OutputLayer(imageWidth * imageHeight * 4, ActivationType.Sigmoid))
            .Build();

        model.Train(x, images, 5, 0.1f);
        model.Save("D:\\mctest.cool");

        //model.Load("D:\\mctest.cool");

        MakeImage(model);
    }

    private static float[][] GetImages(string folderPath, int count)
    {
        float[][] images = new float[count][];
        int index = 0;
        foreach(var file in Directory.EnumerateFiles(folderPath).Take(count))
        {
            images[index++] = GetImagePixel(file);
        }
        return images;
    }

    private static void MakeImage(NNModel model)
    {
        var pred = model.Predict(new float[] { 0.232f }, true);
        Rgba32[] pixelArray = new Rgba32[imageWidth * imageHeight];
        int predIndex = 0;

        for (int y = 0; y < imageHeight; y++)
        {
            for (int x = 0; x < imageWidth; x++)
            {
                pixelArray[y * imageWidth + x] = new Rgba32(pred[predIndex++], pred[predIndex++], pred[predIndex++], pred[predIndex++]);
            }
        }

        using (Image<Rgba32> image = Image.LoadPixelData<Rgba32>(pixelArray, imageWidth, imageHeight))
        {
            image.Save("D:\\test.png");
        }

    }

    private static float[] GetImagePixel(string path)
    {
        float[] pixels = new float[imageWidth * imageHeight * 4];
        int index = 0;
        using Image<Rgba32> image = Image.Load<Rgba32>(path);
        image.ProcessPixelRows(accessor =>
        {
            Rgba32 transparent = Color.Transparent;

            for (int y = 0; y < accessor.Height; y++)
            {
                Span<Rgba32> pixelRow = accessor.GetRowSpan(y);

                for (int x = 0; x < pixelRow.Length; x++)
                {
                    ref Rgba32 pixel = ref pixelRow[x];
                    pixels[index++] = pixel.R;
                    pixels[index++] = pixel.G;
                    pixels[index++] = pixel.B;
                    pixels[index++] = pixel.A > 0 ? 1 : 0;
                }
            }
        });
        return pixels;
    }
}
