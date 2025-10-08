using NNFromScratch.Core;
using System.Runtime.InteropServices;

namespace NNFromScratch.Helper;

internal static class CudaAccel
{
#if DEBUG
    const string DDL_PATH = $"F:\\C#\\NNFromScratch\\x64\\Debug\\CudaWrapper.dll"; //your cuda dll path
#else
    const string DDL_PATH = $"F:\\C#\\NNFromScratch\\x64\\Release\\CudaWrapper.dll"; //your cuda dll path
#endif

    [DllImport(DDL_PATH, CallingConvention = CallingConvention.Cdecl)]
    public static extern void TrainSingle(float[] inputs, float[] desired, int size, float learningRate);

    [DllImport(DDL_PATH, CallingConvention = CallingConvention.Cdecl)]
    public static extern void Init(int totalLayers);

    [DllImport(DDL_PATH, CallingConvention = CallingConvention.Cdecl)]
    public static extern void DoneTraining();
    [DllImport(DDL_PATH, CallingConvention = CallingConvention.Cdecl)]
    [return: MarshalAs(UnmanagedType.I1)]
    public static extern bool CheckCuda();

    [DllImport(DDL_PATH, CallingConvention = CallingConvention.Cdecl)]
    public static extern void Predict(float[] data, float[] prediction);

    [DllImport(DDL_PATH, CallingConvention = CallingConvention.Cdecl)]
    public static extern void Cleanup();

    [DllImport(DDL_PATH, CallingConvention = CallingConvention.Cdecl)]
    public static extern float[] GetOutputNeuronVals();


    [DllImport(DDL_PATH, CallingConvention = CallingConvention.Cdecl)]
    public static extern void InitInputLayer(
        int layerIndex,
        int size,
        float[] biases,
        float[] weights,
        float[] neuronValues,
        float[] errors,
        ActivationType activation);

    [DllImport(DDL_PATH, CallingConvention = CallingConvention.Cdecl)]
    public static extern void InitOutputLayer(
        int layerIndex,
        int prevSize,
        int size,
        float[] biases,
        float[] weights,
        float[] neuronValues,
        float[] errors,
        ActivationType activation);

    [DllImport(DDL_PATH, CallingConvention = CallingConvention.Cdecl)]
    public static extern void InitDenseLayer(
        int layerIndex,
        int prevSize,
        int size,
        float[] biases,
        float[] weights,
        float[] neuronValues,
        float[] errors,
        ActivationType activation);
    [DllImport(DDL_PATH, CallingConvention = CallingConvention.Cdecl)]
    public static extern void TrainFull(
        float[] inputX,
        float[] desired,
        int epochs,
        int samples,
        int features,
        int outputs,
        float learningRate = 0.1f,
        int loggingInterval = 100,
        int epochInterval = 1,
        float evaluatePercent = 10);
}