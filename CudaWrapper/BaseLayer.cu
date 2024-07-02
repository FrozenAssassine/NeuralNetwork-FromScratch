#include "BaseLayer.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "ActivationFunctions.h"

#include <iostream>

/*__global__ void hidden_ErrorWeight(BaseLayer* current, float learningRate, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float err = 0.0f;
        int index = idx * current->previousLayer->Size;

        for (int j = 0; j < current->nextLayer->Size; j++) {
            err += (current->nextLayer->Errors[j] * current->nextLayer->Weights[j * current->Size + idx]);
        }
        float error = err * ActivationFunctions::ActivationDeriv(current->NeuronValues[idx], current->Activation);
        current->Errors[idx] = error;

        error *= learningRate;

        for (int j = 0; j < current->previousLayer->Size; j++) {
            atomicAdd(&current->Weights[index + j], error * current->previousLayer->NeuronValues[j]);
        }
        atomicAdd(&current->Biases[idx], error);
    }
}*/

__global__ void hidden_ErrorWeight(
    float* neuronValues,
    float* prevNeuronValues,
    float* errors,
    float* nextErrors,
    float* weights,
    float* nextWeights,
    float* biases,
    int previousLayerSize,
    int nextLayerSize,
    int size,
    int activationFunction,
    float learningRate
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float err = 0.0f;
        int index = idx * previousLayerSize;

        for (int j = 0; j < nextLayerSize; j++) {
            err += (nextErrors[j] * nextWeights[j * size + idx]);
        }
        float error = err * ActivationFunctions::ActivationDeriv(neuronValues[idx], activationFunction);
        errors[idx] = error;

        error *= learningRate;

        for (int j = 0; j < previousLayerSize; j++) {
            atomicAdd(&weights[index + j], error * prevNeuronValues[j]);
        }
        atomicAdd(&biases[idx], error);
    }
}

/* __global__ void ff_hiddenValues2(BaseLayer* current, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float sum = 0.0f;
        int index = idx * current->previousLayer->Size;
        for (int j = 0; j < current->previousLayer->Size; j++) {
            sum += current->previousLayer->NeuronValues[j] * current->Weights[index + j];
        }
        current->NeuronValues[idx] = ActivationFunctions::Activation(sum + current->Biases[idx], current->Activation);
    }
}*/

__global__ void ff_hiddenValues(int prevSize, float * prevNeuronVal, float * curWeights, float * curNeuronVal, float * curBiases, int curActivation, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float sum = 0.0f;
        int index = idx * prevSize;
        for (int j = 0; j < prevSize; j++) {
            sum += prevNeuronVal[j] * curWeights[index + j];
        }
        curNeuronVal[idx] = ActivationFunctions::Activation(sum + curBiases[idx], curActivation);
    }
}

void BaseLayer::FeedForward(int threadsPerBlock) {

    int blocks = (this->Size + threadsPerBlock - 1) / threadsPerBlock;
    ff_hiddenValues << < blocks, threadsPerBlock >> > (
        this->previousLayer->Size,
        this->previousLayer->NeuronValues,
        this->Weights, 
        this->NeuronValues,
        this->Biases, 
        this->Activation,
        this->Size
        );
}

void BaseLayer::Train(int threadsPerBlock, float* desiredValues, float learningRate) {
    int errorBlocks = (this->Size + threadsPerBlock - 1) / threadsPerBlock;

    hidden_ErrorWeight << < errorBlocks, threadsPerBlock >> > (
        this->NeuronValues, 
        this->previousLayer->NeuronValues,
        this->Errors,
        this->nextLayer->Errors,
        this->Weights,
        this->nextLayer->Weights,
        this->Biases, 
        this->previousLayer->Size, 
        this->nextLayer->Size, 
        this->Size, 
        this->Activation, 
        learningRate
        );
}