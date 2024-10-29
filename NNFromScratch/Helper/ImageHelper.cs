using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp;
using System.Diagnostics;

namespace NNFromScratch.Helper
{
    public class ImageHelper
    {
        public static float[][] GetImages(string folderPath, int count, int width, int height, int start = 0)
        {
            float[][] images = new float[count][];
            int index = 0;
            foreach (var file in Directory.EnumerateFiles(folderPath).Skip(start).Take(count))
            {
                Console.WriteLine("FILE: " + file);
                images[index++] = GetImagePixel_RGB(file, width, height);
            }
            return images;
        }

        public static float[] GetImagePixel_RGB(string path, int width, int height)
        {
            int index = 0;
            using Image<Rgba32> image = Image.Load<Rgba32>(path);
            
            float[] pixels = new float[width * height * 3];

            image.ProcessPixelRows(accessor =>
            {
                Rgba32 transparent = Color.Transparent;

                for (int y = 0; y < accessor.Height; y++)
                {
                    Span<Rgba32> pixelRow = accessor.GetRowSpan(y);

                    for (int x = 0; x < pixelRow.Length; x++)
                    {
                        ref Rgba32 pixel = ref pixelRow[x];
                        pixels[index++] = pixel.R / 255.0f;
                        pixels[index++] = pixel.G / 255.0f;
                        pixels[index++] = pixel.B / 255.0f;
                    }
                }
            });
            return pixels;
        }
    }
}
