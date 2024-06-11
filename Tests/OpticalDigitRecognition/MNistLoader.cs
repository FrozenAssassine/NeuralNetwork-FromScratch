namespace Tests.TestODR;

public static class MNistLoader
{
    public static (float[][] x, float[][] y, int imageWidth, int imageHeight) LoadFromFile(string imageFile, string labelFile)
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
            throw new Exception("Files do not have the same amount of data!");

        int imageWidth = imageReader.ReadInt32MSB();
        int imageHeight = imageReader.ReadInt32MSB();

        float[][] x = new float[imageItemCount][];
        float[][] y = new float[imageItemCount][];

        for (int i = 0; i < imageItemCount; i++)
        {
            byte[] buffer = new byte[imageWidth * imageHeight];
            imageReader.Read(buffer, 0, buffer.Length);
            int digit = labelReader.ReadByte();

            //image pixel data:
            var item = x[i] = new float[buffer.Length];
            for (int j = 0; j < buffer.Length; j++)
            {
                item[j] = buffer[j] / 255.0f;
            }

            //desired image data
            float[] res = new float[10];
            for (int j = 0; j < res.Length; j++)
            {
                if (j == digit)
                    res[j] = 1;
                else
                    res[j] = 0;
            }
            y[i] = res;
        }
        return (x, y, imageWidth, imageHeight);
    }
    private static int ReadInt32MSB(this BinaryReader br)
    {
        byte[] bytes = br.ReadBytes(4);
        return BitConverter.ToInt32(new byte[] { bytes[3], bytes[2], bytes[1], bytes[0] }, 0);
    }
}
