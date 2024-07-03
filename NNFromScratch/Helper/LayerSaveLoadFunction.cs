using NNFromScratch.Core.Layers;

namespace NNFromScratch.Helper;

internal class LayerSaveLoadFunction
{
    public static void Save(BaseLayer layer, BinaryWriter bw)
    {
        if (layer.Weights == null)
            return;

        bw.Write(layer.Biases.Length);
        for (int i = 0; i < layer.Biases.Length; i++)
        {
            bw.Write((double)layer.Biases[i]);
        }

        bw.Write(layer.Weights.Length);

        for (int i = 0; i < layer.Weights.Length; i++)
        {
            bw.Write((double)layer.Weights[i]);
        }
    }
    public static void Load(BaseLayer layer, BinaryReader br)
    {
        int length = br.ReadInt32();
        if (length != layer.Biases.Length)
            throw new InvalidOperationException("Weight data isn't made for this network!");
        for (int i = 0; i < length; i++)
        {
            layer.Biases[i] = (float)br.ReadDouble();
        }
        length = br.ReadInt32();
        if (length != layer.Weights.Length)
            throw new InvalidOperationException("Weight data isn't made for this network!");
        for (int i = 0; i < length; i++)
        {
            layer.Weights[i] = (float)br.ReadDouble();
        }
    }
}
