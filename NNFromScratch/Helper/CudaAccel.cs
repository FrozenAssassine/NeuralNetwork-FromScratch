using System.Runtime.InteropServices;

namespace NNFromScratch.Helper;

internal static class CudaAccel
{
    const string DDL_PATH = "F:\\C#\\NNFromScratch\\x64\\Release\\CudaWrapper.dll"; //your cuda dll path

    [DllImport(DDL_PATH, CallingConvention = CallingConvention.Cdecl)]
    public static extern void InitLayer(int layerIndex, int prevSize, int size, float[] biases, float[] weights, float[] values, float[] errors);

    [DllImport(DDL_PATH, CallingConvention = CallingConvention.Cdecl)]
    public static extern void Init(int totalLayer);

    [DllImport(DDL_PATH, CallingConvention = CallingConvention.Cdecl)]
    public static extern void Train(float[] inputs, float[] desiredOutputs, int size, float learningRate);

    [DllImport(DDL_PATH, CallingConvention = CallingConvention.Cdecl)]
    public static extern void Predict(float[] data, float[] prediction, int n);
    
    [DllImport(DDL_PATH, CallingConvention = CallingConvention.Cdecl)]
    public static extern void DoneTraining();
    [DllImport(DDL_PATH, CallingConvention = CallingConvention.Cdecl)]
    public static extern bool CheckCuda();
}
