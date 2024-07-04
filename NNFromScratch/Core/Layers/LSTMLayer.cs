using NNFromScratch.Helper;

namespace NNFromScratch.Core.Layers;

public class LSTMLayer : BaseLayer
{
    private float[] CellState;
    private float[] OutputGate;
    private float[] ForgetGate;
    private float[] InputGate;
    private float[] CandidateCellState;

    private float[] WeightsInput;
    private float[] WeightsForget;
    private float[] WeightsOutput;
    private float[] WeightsCandidate;

    float[] outputGradients;
    float[] cellStateGradients;
    float[] inputGateGradients;
    float[] forgetGateGradients;
    float[] candidateCellGradients;

    public LSTMLayer(int size)
    {
        Size = size;
    }

    public override void FeedForward()
    {
        Parallel.For(0, Size, (idx) =>
        {
            float inputGateSum = Biases[idx];
            float forgetGateSum = Biases[idx];
            float outputGateSum = Biases[idx];
            float candidateCellSum = Biases[idx];

            for (int j = 0; j < PreviousLayer.Size; j++)
            {
                inputGateSum += PreviousLayer.NeuronValues[j] * WeightsInput[j * Size + idx];
                forgetGateSum += PreviousLayer.NeuronValues[j] * WeightsForget[j * Size + idx];
                outputGateSum += PreviousLayer.NeuronValues[j] * WeightsOutput[j * Size + idx];
                candidateCellSum += PreviousLayer.NeuronValues[j] * WeightsCandidate[j * Size + idx];
            }

            InputGate[idx] = ActivationFunctions.Activation(inputGateSum, ActivationType.Sigmoid);
            ForgetGate[idx] = ActivationFunctions.Activation(forgetGateSum, ActivationType.Sigmoid);
            OutputGate[idx] = ActivationFunctions.Activation(outputGateSum, ActivationType.Sigmoid);
            CandidateCellState[idx] = ActivationFunctions.Activation(candidateCellSum, ActivationType.TanH);

            CellState[idx] = CellState[idx] * ForgetGate[idx] + InputGate[idx] * CandidateCellState[idx];
            NeuronValues[idx] = ActivationFunctions.Activation(CellState[idx], ActivationType.TanH) * OutputGate[idx];
        });
    }

    public override void Train(float[] desiredValues, float learningRate)
    {
        // Calculate the error at the output layer
        for (int i = 0; i < Size; i++)
        {
            Errors[i] = i < desiredValues.Length ? desiredValues[i] - NeuronValues[i] : 0; // Only calculate error for indices up to VocabularySize
            outputGradients[i] = Errors[i] * ActivationFunctions.Activation(CellState[i], ActivationType.TanH) * OutputGate[i] * (1 - OutputGate[i]);
        }

        // Backpropagate through the LSTM gates and cell state
        Parallel.For(0, Size, (idx) =>
        {
            cellStateGradients[idx] = Errors[idx] * OutputGate[idx] * ActivationFunctions.ActivationDeriv(CellState[idx], ActivationType.TanH);

            inputGateGradients[idx] = cellStateGradients[idx] * CandidateCellState[idx] * InputGate[idx] * (1 - InputGate[idx]);
            forgetGateGradients[idx] = cellStateGradients[idx] * CellState[idx] * ForgetGate[idx] * (1 - ForgetGate[idx]);
            candidateCellGradients[idx] = cellStateGradients[idx] * InputGate[idx] * ActivationFunctions.ActivationDeriv(CandidateCellState[idx], ActivationType.TanH);
        });

        // Update weights and biases
        Parallel.For(0, Size, (idx) =>
        {
            for (int j = 0; j < PreviousLayer.Size; j++)
            {
                WeightsInput[j * Size + idx] += learningRate * inputGateGradients[idx] * PreviousLayer.NeuronValues[j];
                WeightsForget[j * Size + idx] += learningRate * forgetGateGradients[idx] * PreviousLayer.NeuronValues[j];
                WeightsOutput[j * Size + idx] += learningRate * outputGradients[idx] * PreviousLayer.NeuronValues[j];
                WeightsCandidate[j * Size + idx] += learningRate * candidateCellGradients[idx] * PreviousLayer.NeuronValues[j];
            }

            Biases[idx] += learningRate * Errors[idx];
        });

        // Update the errors for the previous layer
        for (int i = 0; i < PreviousLayer.Size; i++)
        {
            PreviousLayer.Errors[i] = 0;
            for (int j = 0; j < Size; j++)
            {
                PreviousLayer.Errors[i] += inputGateGradients[j] * WeightsInput[i * Size + j];
                PreviousLayer.Errors[i] += forgetGateGradients[j] * WeightsForget[i * Size + j];
                PreviousLayer.Errors[i] += outputGradients[j] * WeightsOutput[i * Size + j];
                PreviousLayer.Errors[i] += candidateCellGradients[j] * WeightsCandidate[i * Size + j];
            }
        }
    }

    public override void Summary()
    {
        Console.WriteLine($"LSTM Layer - Size: {Size}");
    }

    public override void Save(BinaryWriter bw)
    {
        bw.Write(Size);
        bw.Write(Biases.Length);
        foreach (var bias in Biases) bw.Write(bias);

        bw.Write(WeightsInput.Length);
        foreach (var weight in WeightsInput) bw.Write(weight);

        bw.Write(WeightsForget.Length);
        foreach (var weight in WeightsForget) bw.Write(weight);

        bw.Write(WeightsOutput.Length);
        foreach (var weight in WeightsOutput) bw.Write(weight);

        bw.Write(WeightsCandidate.Length);
        foreach (var weight in WeightsCandidate) bw.Write(weight);
    }

    public override void Load(BinaryReader br)
    {
        Size = br.ReadInt32();
        int biasesLength = br.ReadInt32();
        Biases = new float[biasesLength];
        for (int i = 0; i < biasesLength; i++) Biases[i] = br.ReadSingle();

        int weightsInputLength = br.ReadInt32();
        WeightsInput = new float[weightsInputLength];
        for (int i = 0; i < weightsInputLength; i++) WeightsInput[i] = br.ReadSingle();

        int weightsForgetLength = br.ReadInt32();
        WeightsForget = new float[weightsForgetLength];
        for (int i = 0; i < weightsForgetLength; i++) WeightsForget[i] = br.ReadSingle();

        int weightsOutputLength = br.ReadInt32();
        WeightsOutput = new float[weightsOutputLength];
        for (int i = 0; i < weightsOutputLength; i++) WeightsOutput[i] = br.ReadSingle();

        int weightsCandidateLength = br.ReadInt32();
        WeightsCandidate = new float[weightsCandidateLength];
        for (int i = 0; i < weightsCandidateLength; i++) WeightsCandidate[i] = br.ReadSingle();
    }

    public override void Initialize()
    {
        Biases = new float[this.Size];
        NeuronValues = new float[this.Size];
        Errors = new float[this.Size];

        CellState = new float[this.Size];
        OutputGate = new float[this.Size];
        ForgetGate = new float[this.Size];
        InputGate = new float[this.Size];
        CandidateCellState = new float[this.Size];

        int weightsSize = this.PreviousLayer.Size * this.Size;
        WeightsInput = new float[weightsSize];
        WeightsForget = new float[weightsSize];
        WeightsOutput = new float[weightsSize];
        WeightsCandidate = new float[weightsSize];

        outputGradients = new float[Size];
        cellStateGradients = new float[Size];
        inputGateGradients = new float[Size];
        forgetGateGradients = new float[Size];
        candidateCellGradients = new float[Size];

        for (int i = 0; i < Biases.Length; i++) Biases[i] = MathHelper.RandomFloat1_1();

        for (int i = 0; i < WeightsInput.Length; i++) WeightsInput[i] = MathHelper.RandomFloat1_1();
        for (int i = 0; i < WeightsForget.Length; i++) WeightsForget[i] = MathHelper.RandomFloat1_1();
        for (int i = 0; i < WeightsOutput.Length; i++) WeightsOutput[i] = MathHelper.RandomFloat1_1();
        for (int i = 0; i < WeightsCandidate.Length; i++) WeightsCandidate[i] = MathHelper.RandomFloat1_1();
    }

    public override void InitializeCuda(int index)
    {
        CudaAccel.InitLSTMLayer(
            index,
            PreviousLayer.Size,
            this.Size,
            this.Biases,
            this.NeuronValues,
            this.Errors,
            this.WeightsInput,
            this.WeightsForget,
            this.WeightsOutput,
            this.WeightsCandidate,
            this.CellState,
            this.OutputGate,
            this.ForgetGate,
            this.InputGate,
            this.CandidateCellState,
            this.inputGateGradients,
            this.forgetGateGradients,
            this.outputGradients,
            this.candidateCellGradients
            );
    }
}