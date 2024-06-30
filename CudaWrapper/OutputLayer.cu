#include "OutputLayer.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "ActivationFunctions.h"
#include <iostream>

__global__ void ff_outputValues(BaseLayer* outputLayer, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float sum = 0.0f;
        int weightIndex = idx * outputLayer->previousLayer->Size;
        for (int j = 0; j < outputLayer->Size; j++) {
            sum += outputLayer->NeuronValues[j] * outputLayer->Weights[weightIndex + j];
        }
        outputLayer->NeuronValues[idx] = ActivationFunctions::Activation(sum + outputLayer->Biases[idx], outputLayer->Activation);
    }
}

__global__ void output_Errors(BaseLayer* output, float* desired, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output->Errors[idx] = desired[idx] - output->NeuronValues[idx];
    }
}

__global__ void output_WeightsBiases(BaseLayer* output, float learningRate, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float derivNeuronVal = learningRate * output->Errors[idx] * ActivationFunctions::ActivationDeriv(output->NeuronValues[idx], output->Activation);
        int weightIndex = idx * output->previousLayer->Size;

        for (int j = 0; j < output->previousLayer->Size; j++) {
            atomicAdd(&output->Weights[weightIndex + j], derivNeuronVal * output->previousLayer->NeuronValues[j]);
        }
        atomicAdd(&output->Biases[idx], learningRate * output->Errors[idx] * ActivationFunctions::ActivationDeriv(output->NeuronValues[idx], output->Activation));
    }
}


void OutputLayer::FeedForward(int threadsPerBlock) {

    //Compute neuron values for output layer
    int blocks = (this->Size + threadsPerBlock - 1) / threadsPerBlock;
    ff_outputValues << <blocks, threadsPerBlock >> > (this, this->Size);
}

void OutputLayer::Train(int threadsPerBlock, float * desiredValues, float learningRate) {
    printf("Train output");

    // Calculate errors for the output layer
    int outputBlocks = (this->Size + threadsPerBlock - 1) / threadsPerBlock;
    output_Errors << <outputBlocks, threadsPerBlock >> > (this, desiredValues, this->Size);
     

    // Update weights and biases for the output layer
    output_WeightsBiases << <outputBlocks, threadsPerBlock >> > (
        this,
        learningRate,
        this->Size
        );
}