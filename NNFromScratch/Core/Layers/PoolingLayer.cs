using System;

namespace NNFromScratch.Core.Layers;

public class PoolingLayer : BaseLayer
{
    public int PoolSize { get; }
    public int Stride { get; }
    public int inputHeight;
    public int inputWidth;

    public PoolingLayer(int inputWidth, int inputHeight, int poolSize, int stride)
    {
        this.PoolSize = poolSize;
        this.Stride = stride;
        this.inputWidth = inputWidth;
        this.inputHeight = inputHeight;
    }

    public int CalculateDenseLayerNeurons(int featureMapX, int featureMapY, int depth)
    {
        int outputWidth = (featureMapX - PoolSize) / Stride + 1;
        int outputHeight = (featureMapY - PoolSize) / Stride + 1;

        return outputWidth * outputHeight * depth;
    }

    public override void FeedForward()
    {
        if (this.PreviousLayer is not ConvolutionalLayer convLayer)
        {
            throw new Exception("Previous layer has to be ConvolutionalLayer");
        }

        int outputWidth = (inputWidth - PoolSize) / Stride + 1;
        int outputHeight = (inputHeight - PoolSize) / Stride + 1;
        float[] output = new float[outputWidth * outputHeight];

        Parallel.For(0, outputHeight, (idy) => //for (int y = 0; y < outputHeight; y++)
        {
            for (int x = 0; x < outputWidth; x++)
            {
                float sum = 0;
                int count = 0;

                for (int j = 0; j < PoolSize; j++)
                {
                    for (int i = 0; i < PoolSize; i++)
                    {
                        int inputX = x * Stride + i;
                        int inputY = idy * Stride + j;

                        if (inputX < inputWidth && inputY < inputHeight)
                        {
                            sum += convLayer.featureMap[inputY * inputWidth + inputX];
                            count++;
                        }
                    }
                }

                output[idy * outputWidth + x] = sum / count;
            }
        });

        this.NeuronValues = output; //set for dense layer
    }

    public override void Initialize()
    {

    }

    public override void InitializeCuda(int index)
    {
        throw new NotImplementedException();
    }

    public override void Load(BinaryReader br)
    {
        throw new NotImplementedException();
    }

    public override void Save(BinaryWriter bw)
    {
        throw new NotImplementedException();
    }

    public override void Summary()
    {
        throw new NotImplementedException();
    }

    public override void Train(float[] desiredValues, float learningRate)
    {
        //nothing to implement here:
    }
}
