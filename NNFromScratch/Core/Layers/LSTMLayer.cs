using NNFromScratch.Helper;
using System.Numerics;

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
    float[] PreviousCellState;
    float[] PreviousNeuronValues;

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

            CellState[idx] = PreviousCellState[idx] * ForgetGate[idx] + InputGate[idx] * CandidateCellState[idx];
            NeuronValues[idx] = ActivationFunctions.Activation(CellState[idx], ActivationType.TanH) * OutputGate[idx];

            // Store current states for next time step
            PreviousCellState[idx] = CellState[idx];
            PreviousNeuronValues[idx] = NeuronValues[idx];
        });
    }

    public override void Train(float[] desiredValues, float learningRate)
    {
        if (NextLayer == null)
        {
            for (int i = 0; i < Size; i++)
            {
                Errors[i] = i < desiredValues.Length ? desiredValues[i] - NeuronValues[i] : 0;
            }
        }
        else
        {
            // If there is a next layer, propagate errors from the next layer
            for (int i = 0; i < Size; i++)
            {
                Errors[i] = NextLayer.Errors[i]; // Use the propagated errors
            }
        }

        // Backpropagate through the LSTM gates and cell state
        Parallel.For(0, Size, (idx) =>
        {
            float tanhCellState = ActivationFunctions.Activation(CellState[idx], ActivationType.TanH);
            outputGradients[idx] = Errors[idx] * tanhCellState * ActivationFunctions.ActivationDeriv(OutputGate[idx], ActivationType.Sigmoid);

            cellStateGradients[idx] = Errors[idx] * OutputGate[idx] * ActivationFunctions.ActivationDeriv(CellState[idx], ActivationType.TanH);

            inputGateGradients[idx] = cellStateGradients[idx] * CandidateCellState[idx] * ActivationFunctions.ActivationDeriv(InputGate[idx], ActivationType.Sigmoid);
            forgetGateGradients[idx] = cellStateGradients[idx] * CellState[idx] * ActivationFunctions.ActivationDeriv(ForgetGate[idx], ActivationType.Sigmoid);
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

            Biases[idx] += learningRate * (
                inputGateGradients[idx] + forgetGateGradients[idx] +
                outputGradients[idx] + candidateCellGradients[idx]
            );
        });

        // Update the errors for the previous layer
        Parallel.For(0, PreviousLayer.Size, (idx) =>
        {
            PreviousLayer.Errors[idx] = 0;
            for (int j = 0; j < Size; j++)
            {
                PreviousLayer.Errors[idx] += inputGateGradients[j] * WeightsInput[idx * Size + j];
                PreviousLayer.Errors[idx] += forgetGateGradients[j] * WeightsForget[idx * Size + j];
                PreviousLayer.Errors[idx] += outputGradients[j] * WeightsOutput[idx * Size + j];
                PreviousLayer.Errors[idx] += candidateCellGradients[j] * WeightsCandidate[idx * Size + j];
            }
        });
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

        PreviousCellState = new float[this.Size];
        PreviousNeuronValues = new float[this.Size];

        for (int i = 0; i < Biases.Length; i++) Biases[i] = MathHelper.RandomFloat1_1();

        for (int i = 0; i < WeightsInput.Length; i++) WeightsInput[i] = MathHelper.RandomFloat1_1();
        for (int i = 0; i < WeightsForget.Length; i++) WeightsForget[i] = MathHelper.RandomFloat1_1();
        for (int i = 0; i < WeightsOutput.Length; i++) WeightsOutput[i] = MathHelper.RandomFloat1_1();
        for (int i = 0; i < WeightsCandidate.Length; i++) WeightsCandidate[i] = MathHelper.RandomFloat1_1();
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