using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Test1.Data1;

public static class MNistLoader
{
    public static DigitData[] LoadFromFile(string imageFile, string labelFile)
    {
        MemoryStream imageData = new MemoryStream(File.ReadAllBytes(imageFile));
        MemoryStream labelData = new MemoryStream(File.ReadAllBytes(labelFile));
        BinaryReader imageReader = new BinaryReader(imageData);
        BinaryReader labelReader = new BinaryReader(labelData);
        int imageMagicNumber = imageReader.ReadInt32MSB();
        int labelMagicNumber = labelReader.ReadInt32MSB();
        int imageItemCount = imageReader.ReadInt32MSB();
        int labelItemCount = labelReader.ReadInt32MSB();
        if (imageItemCount != labelItemCount)
            throw new Exception("Files doesn't have the same amount of data!");
        int imageWidth = imageReader.ReadInt32MSB();
        int imageHeight = imageReader.ReadInt32MSB();
        DigitData[] digits = new DigitData[imageItemCount];
        for (int i = 0; i < imageItemCount; i++)
        {
            byte[] buffer = new byte[imageWidth * imageHeight];
            imageReader.Read(buffer, 0, buffer.Length);
            int digit = labelReader.ReadByte();
            digits[i] = new DigitData()
            {
                Data = buffer,
                Digit = digit,
                Width = imageWidth,
                Height = imageHeight
            };
        }
        return digits;
    }

    private static int ReadInt32MSB(this BinaryReader br)
    {
        byte[] bytes = br.ReadBytes(4);
        return BitConverter.ToInt32(new byte[] { bytes[3], bytes[2], bytes[1], bytes[0] }, 0);
    }
}
