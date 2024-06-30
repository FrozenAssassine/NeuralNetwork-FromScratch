#include "BaseLayer.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "ActivationFunctions.h"

#include <iostream>

__global__ void hidden_ErrorWeight(BaseLayer* current, float learningRate, int size) {
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
}

__global__ void ff_hiddenValues(BaseLayer* current, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float sum = 0.0f;
        int index = idx * current->previousLayer->Size;
        for (int j = 0; j < current->previousLayer->Size; j++) {
            sum += current->previousLayer->NeuronValues[j] * current->Weights[index + j];
        }
        current->NeuronValues[idx] = ActivationFunctions::Activation(sum + current->Biases[idx], current->Activation);
    }
}

void BaseLayer::FeedForward(int threadsPerBlock) {

    int blocks = (this->Size + threadsPerBlock - 1) / threadsPerBlock;
    ff_hiddenValues << < blocks, threadsPerBlock >> > (this, this->Size);
}

void BaseLayer::Train(int threadsPerBlock, float* desiredValues, float learningRate) {
    printf("Train Base");

    int errorBlocks = (this->Size + threadsPerBlock - 1) / threadsPerBlock;
    hidden_ErrorWeight << <errorBlocks, threadsPerBlock >> > (this, learningRate, this->Size);
}