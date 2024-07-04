#include "LSTMLayer.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "ActivationFunctions.h"

__global__ void WeightsBiases(
    float* WeightsInput,
    float* WeightsForget,
    float* WeightsOutput,
    float* WeightsCandidate,
    float* inputGateGradients,
    float* forgetGateGradients,
    float* outputGradients,
    float* candidateCellGradients,
    float* Errors,
    float* Biases,
    float* prevNeuronValues,
    float learningRate,
    int prevSize,
    int size
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        for (int j = 0; j < prevSize; j++)
        {
            WeightsInput[j * size + idx] += learningRate * inputGateGradients[idx] * prevNeuronValues[j];
            WeightsForget[j * size + idx] += learningRate * forgetGateGradients[idx] * prevNeuronValues[j];
            WeightsOutput[j * size + idx] += learningRate * outputGradients[idx] * prevNeuronValues[j];
            WeightsCandidate[j * size + idx] += learningRate * candidateCellGradients[idx] * prevNeuronValues[j];
        }

        Biases[idx] += learningRate * Errors[idx];
    }
}

__global__ void ff_values(
    float* WeightsInput,
    float* WeightsForget,
    float* WeightsOutput,
    float* WeightsCandidate,
    float* InputGate,
    float* ForgetGate,
    float* OutputGate,
    float* CandidateCellState,
    float* CellState,
    float* NeuronValues,
    float* Biases,
    float* prevNeuronValues,
    int size,
    int prevSize
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float inputGateSum = Biases[idx];
        float forgetGateSum = Biases[idx];
        float outputGateSum = Biases[idx];
        float candidateCellSum = Biases[idx];

        for (int j = 0; j < prevSize; j++)
        {
            inputGateSum += prevNeuronValues[j] * WeightsInput[j * size + idx];
            forgetGateSum += prevNeuronValues[j] * WeightsForget[j * size + idx];
            outputGateSum += prevNeuronValues[j] * WeightsOutput[j * size + idx];
            candidateCellSum += prevNeuronValues[j] * WeightsCandidate[j * size + idx];
        }

        InputGate[idx] = ActivationFunctions::Activation(inputGateSum, 0); //sigmoid
        ForgetGate[idx] = ActivationFunctions::Activation(forgetGateSum, 0); //sigmoid
        OutputGate[idx] = ActivationFunctions::Activation(outputGateSum, 0); //sigmoid
        CandidateCellState[idx] = ActivationFunctions::Activation(candidateCellSum, 3); //tanh

        CellState[idx] = CellState[idx] * ForgetGate[idx] + InputGate[idx] * CandidateCellState[idx];
        NeuronValues[idx] = ActivationFunctions::Activation(CellState[idx], 3) * OutputGate[idx]; //tanh
    }
}

__global__ void backPropagate(
    float* cellStateGradients,
    float* Errors,
    float* OutputGate,
    float* CellState,
    float* CandidateCellState,
    float* inputGateGradients,
    float* candidateCellGradients,
    float* forgetGateGradients,
    float* ForgetGate,
    float* InputGate,
    int size
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        cellStateGradients[idx] = Errors[idx] * OutputGate[idx] * ActivationFunctions::ActivationDeriv(CellState[idx], 3);

        inputGateGradients[idx] = cellStateGradients[idx] * CandidateCellState[idx] * InputGate[idx] * (1 - InputGate[idx]);
        forgetGateGradients[idx] = cellStateGradients[idx] * CellState[idx] * ForgetGate[idx] * (1 - ForgetGate[idx]);
        candidateCellGradients[idx] = cellStateGradients[idx] * InputGate[idx] * ActivationFunctions::ActivationDeriv(CandidateCellState[idx], 3);
    }
}

__global__ void calcOutputErrors(
    float* Errors,
    float* desiredValues,
    float* NeuronValues,
    float* CellState,
    float* outputGradients,
    float* OutputGate,
    int desiredLength,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        Errors[idx] = idx < desiredLength ? desiredValues[idx] - NeuronValues[idx] : 0;
        outputGradients[idx] = Errors[idx] * ActivationFunctions::Activation(CellState[idx], 3) * OutputGate[idx] * (1 - OutputGate[idx]);
    }
}

void LSTMLayer::FeedForward(int threadsPerBlock)
{
    int blocks = (this->Size + threadsPerBlock - 1) / threadsPerBlock;

    ff_values << < blocks, threadsPerBlock >> > (
        this->WeightsInput,
        this->WeightsForget,
        this->WeightsOutput,
        this->WeightsCandidate,
        this->InputGate,
        this->ForgetGate,
        this->OutputGate,
        this->CandidateCellState,
        this->CellState,
        this->NeuronValues,
        this->Biases,
        this->previousLayer->NeuronValues,
        this->Size,
        this->previousLayer->Size
        );
}

void LSTMLayer::Train(int threadsPerBlock, float* desiredValues, float learningRate)
{
    int desiredLength = sizeof(desiredValues) * sizeof(float);

    // Calculate the error at the output layer
    int blocks = (this->Size + threadsPerBlock - 1) / threadsPerBlock;
    calcOutputErrors<<<blocks, threadsPerBlock >>>(
        Errors,
        desiredValues,
        NeuronValues,
        CellState,
        outputGradients,
        OutputGate,
        desiredLength,
        Size
    );

    // Backpropagate through the LSTM gates and cell state
    backPropagate << <blocks, threadsPerBlock >> > (
        this->cellStateGradients,
        this->Errors,
        this->OutputGate,
        this->CellState,
        this->CandidateCellState,
        this->inputGateGradients,
        this->candidateCellGradients,
        this->forgetGateGradients,
        this->ForgetGate,
        this->InputGate,
        this->Size
        );


    // Update weights and biases
    WeightsBiases << <blocks, threadsPerBlock >> > (
        WeightsInput,
        WeightsForget,
        WeightsOutput,
        WeightsCandidate,
        inputGateGradients,
        forgetGateGradients,
        outputGradients,
        candidateCellGradients,
        Errors,
        Biases,
        previousLayer->NeuronValues,
        learningRate,
        previousLayer->Size,
        Size
        );

    //for (int i = 0; i < this->previousLayer->Size; i++)
    //{
    //    previousLayer->Errors[i] = 0;
    //    for (int j = 0; j < Size; j++)
    //    {
    //        previousLayer->Errors[i] += inputGateGradients[j] * WeightsInput[i * Size + j];
    //        previousLayer->Errors[i] += forgetGateGradients[j] * WeightsForget[i * Size + j];
    //        previousLayer->Errors[i] += outputGradients[j] * WeightsOutput[i * Size + j];
    //        previousLayer->Errors[i] += candidateCellGradients[j] * WeightsCandidate[i * Size + j];
    //    }
    //}
}