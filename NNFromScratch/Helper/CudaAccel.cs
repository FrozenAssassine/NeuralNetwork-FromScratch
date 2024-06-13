using System.Runtime.InteropServices;

namespace NNFromScratch.Helper;

internal static class CudaAccel
{
    [DllImport("CudaC#Wrapper.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern void InitLayer(int layerIndex, int prevSize, int size, float[] biases, float[] weights, float[] values, float[] errors);

    [DllImport("CudaC#Wrapper.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern void Init(int totalLayer);

    [DllImport("CudaC#Wrapper.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern void Train(float[] inputs, float[] desiredOutputs, int size, float learningRate);

    [DllImport("CudaC#Wrapper.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern float[] FeedForward(float[] data, int n);
}
