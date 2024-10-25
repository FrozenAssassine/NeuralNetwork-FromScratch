namespace NNFromScratch.Core.Layers;

public class PoolingLayer : BaseLayer
{
    public int PoolSize { get; }
    public int Stride { get; }
    public int inputHeight;
    public int inputWidth;
    private int pooledHeight;
    private int pooledWidth;
    
    public PoolingLayer(int inputWidth, int inputHeight, int featureMapX, int featureMapY, int poolSize, int stride)
    {
        this.PoolSize = poolSize;
        this.Stride = stride;
        this.inputWidth = inputWidth;
        this.inputHeight = inputHeight;

        //calculate the output dimensions:
        this.pooledWidth = (featureMapX - PoolSize) / Stride + 1;
        this.pooledHeight = (featureMapY - PoolSize) / Stride + 1;
        this.Size = pooledWidth * pooledHeight;
    }

    public override void FeedForward()
    {
        if (this.PreviousLayer is not ConvolutionalLayer convLayer)
        {
            throw new Exception("Previous layer has to be ConvolutionalLayer");
        }

        Parallel.For(0, pooledHeight, (idy) => //for (int y = 0; y < outputHeight; y++)
        {
            for (int x = 0; x < pooledWidth; x++)
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

                this.NeuronValues[idy * pooledWidth + x] = sum / count;
            }
        });
    }

    public override void Initialize()
    {
        this.NeuronValues = new float[this.Size];
    }

    public override void InitializeCuda(int index)
    {
        throw new NotImplementedException();
    }

    public override void Load(BinaryReader br)
    {
        //nothing to load here
    }

    public override void Save(BinaryWriter bw)
    {
        //nothing to save here
    }

    public override void Summary()
    {
        Console.WriteLine($"Pooling Layer of {this.Size} and outputSize of ({this.pooledWidth}x{this.pooledHeight})");
    }

    public override void Train(float[] desiredValues, float learningRate)
    {
        //nothing to implement here:
    }
}
