using System.Runtime.InteropServices;

namespace NNFromScratch.Helper;

internal static class CudaAccel
{
    [DllImport("CudaC#Wrapper.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern void InitLayer(int layerIndex,
        float[] biases, float[] neuronValues, float[] errors,
        float[] weights, float size);

    [DllImport("CudaC#Wrapper.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern void Init(int totalLayer);

    [DllImport("CudaC#Wrapper.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern void Train(float[] inputs, float[] desiredOutputs, int size, float learningRate);
}
