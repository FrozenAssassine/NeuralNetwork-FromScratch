using NNFromScratch.Helper;

namespace NNFromScratch.Core.Layers;

public class OutputLayer : BaseLayer
{
    public OutputLayer(int size, ActivationType activation = ActivationType.Sigmoid)
    {
        this.Size = size;
        this.ActivationFunction = activation;
    }

    public override void Load(BinaryReader br)
    {
        LayerSaveLoadFunction.Load(this, br);
    }

    public override void Save(BinaryWriter bw)
    {
        LayerSaveLoadFunction.Save(this, bw);
    }

    public override void Summary()
    {
        Console.WriteLine($"Output Layer of {Size} Neurons and {Weights.Length} Weights");
    }

    public override void FeedForward()
    {
        Parallel.For(0, this.Size, (idx) =>
        {
            float sum = 0.0f;
            int weightIndex = idx * this.PreviousLayer.Size;
            for (int j = 0; j < this.PreviousLayer.Size; j++)
            {
                sum += this.PreviousLayer.NeuronValues[j] * this.Weights[weightIndex + j];
            }
            this.NeuronValues[idx] = ActivationFunctions.Activation(sum + this.Biases[idx], this.ActivationFunction);
        });
    }

    public override void Train(float[] desiredValues, float learningRate)
    {
        if(desiredValues.Length != this.Size)
        {
            throw new Exception("Output layer count does not match provided data");
        }

        //output -> error
        Parallel.For(0, this.Size, (idx) =>
        {
            this.Errors[idx] = desiredValues[idx] - this.NeuronValues[idx];
        });

        //output -> weights and biases
        Parallel.For(0, this.Size, (idx) =>
        {
            float derivNeuronVal = learningRate * this.Errors[idx] * ActivationFunctions.ActivationDeriv(this.NeuronValues[idx], this.ActivationFunction);
            int weightIndex = idx * this.PreviousLayer.Size;

            for (int j = 0; j < this.PreviousLayer.Size; j++)
            {
                this.Weights[weightIndex + j] += derivNeuronVal * this.PreviousLayer.NeuronValues[j];
            }
            this.Biases[idx] += learningRate * this.Errors[idx] * ActivationFunctions.ActivationDeriv(this.NeuronValues[idx], this.ActivationFunction);
        });
    }

    public override void Initialize()
    {
        LayerInitialisationHelper.InitializeLayer(this);
    }

    public override void InitializeCuda(int index)
    {
        CudaAccel.InitOutputLayer(index, this.PreviousLayer.Size, this.Size, this.Biases, this.Weights, this.NeuronValues, this.Errors, this.ActivationFunction);
    }
}
