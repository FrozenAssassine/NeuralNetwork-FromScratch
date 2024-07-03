#include "OutputLayer.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "ActivationFunctions.h"
#include <iostream>

/*__global__ void ff_outputValues(DenseLayer* outputLayer, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float sum = 0.0f;
        int weightIndex = idx * outputLayer->previousLayer->Size;
        for (int j = 0; j < outputLayer->Size; j++) {
            sum += outputLayer->NeuronValues[j] * outputLayer->Weights[weightIndex + j];
        }
        outputLayer->NeuronValues[idx] = ActivationFunctions::Activation(sum + outputLayer->Biases[idx], outputLayer->Activation);
    }
}*/

__global__ void ff_outputValues(float* neuronValues, float* prevNeuronValues, float* weights, float* biases, int previousLayerSize, int activationFunction, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float sum = 0.0f;
        int weightIndex = idx * previousLayerSize;
        for (int j = 0; j < previousLayerSize; j++) {
            sum += prevNeuronValues[j] * weights[weightIndex + j];
        }
        neuronValues[idx] = ActivationFunctions::Activation(sum + biases[idx], activationFunction);
    }
}

__global__ void output_Errors(float * errors, float * neuronValues, float* desired, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        errors[idx] = desired[idx] - neuronValues[idx];
    }
}

/*__global__ void output_WeightsBiases(DenseLayer* output, float learningRate, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float derivNeuronVal = learningRate * output->Errors[idx] * ActivationFunctions::ActivationDeriv(output->NeuronValues[idx], output->Activation);
        int weightIndex = idx * output->previousLayer->Size;

        for (int j = 0; j < output->previousLayer->Size; j++) {
            atomicAdd(&output->Weights[weightIndex + j], derivNeuronVal * output->previousLayer->NeuronValues[j]);
        }
        atomicAdd(&output->Biases[idx], learningRate * output->Errors[idx] * ActivationFunctions::ActivationDeriv(output->NeuronValues[idx], output->Activation));
    }
}*/

__global__ void output_WeightsBiases(float* errors, float* neuronValues, float* prevNeuronValues, float* weights, float* biases, int previousLayerSize, int size, int activationFunction, float learningRate) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float derivNeuronVal = learningRate * errors[idx] * ActivationFunctions::ActivationDeriv(neuronValues[idx], activationFunction);
        int weightIndex = idx * previousLayerSize;

        for (int j = 0; j < previousLayerSize; j++) {
            atomicAdd(&weights[weightIndex + j], derivNeuronVal * prevNeuronValues[j]);
        }
        atomicAdd(&biases[idx], learningRate * errors[idx] * ActivationFunctions::ActivationDeriv(neuronValues[idx], activationFunction));
    }
}

void OutputLayer::FeedForward(int threadsPerBlock) {
    //Compute neuron values for output layer
    int blocks = (this->Size + threadsPerBlock - 1) / threadsPerBlock;
    ff_outputValues << <blocks, threadsPerBlock >> > (this->NeuronValues, this->previousLayer->NeuronValues, this->Weights, this->Biases, this->previousLayer->Size, this->Activation, this->Size);
}

void OutputLayer::Train(int threadsPerBlock, float * desiredValues, float learningRate) {

    // Calculate errors for the output layer
    int outputBlocks = (this->Size + threadsPerBlock - 1) / threadsPerBlock;  
    output_Errors << <outputBlocks, threadsPerBlock >> > (
        this->Errors, 
        this->NeuronValues, 
        desiredValues, 
        this->Size
        );
     
    // Update weights and biases for the output layer
    output_WeightsBiases << <outputBlocks, threadsPerBlock >> > (
        this->Errors,
        this->NeuronValues,
        this->previousLayer->NeuronValues,
        this->Weights,
        this->Biases,
        this->previousLayer->Size,
        this->Size,
        this->Activation,
        learningRate
        );
}