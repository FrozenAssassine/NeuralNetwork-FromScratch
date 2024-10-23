namespace NNFromScratch.Helper;

internal class LayerSaveLoadHelper
{
    public static void SaveData(float[] data, BinaryWriter bw)
    {
        if (data == null)
            throw new Exception("Could not save data: Null exception!");

        bw.Write(data.Length);

        for (int i = 0; i < data.Length; i++)
        {
            bw.Write((double)data[i]);
        }
    }

    public static void LoadData(float[] data, BinaryReader br) 
    {
        if (data == null)
            throw new Exception("Could not load data: Null exception!");

        int length = br.ReadInt32();
        if (length != data.Length)
            throw new InvalidOperationException("Weight data isn't made for this network!");
        for (int i = 0; i < length; i++)
        {
            data[i] = (float)br.ReadDouble();
        }
    }
}
