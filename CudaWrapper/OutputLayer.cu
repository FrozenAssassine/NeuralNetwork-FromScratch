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


__global__ void ff_outputValues_Softmax_GPU(
    float* neuronValues,
    float* prevNeuronValues,
    float* weights,
    float* biases,
    int previousLayerSize,
    int size
) {
    extern __shared__ float shared[];
    float* z = shared;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Step 1: Compute pre-activation z_i
    if (idx < size) {
        float sum = 0.0f;
        int weightIndex = idx * previousLayerSize;
        for (int j = 0; j < previousLayerSize; j++) {
            sum += prevNeuronValues[j] * weights[weightIndex + j];
        }
        z[idx] = sum + biases[idx];
    }

    __syncthreads();

    // Step 2: Find max(z)
    float maxVal = -1e20f;
    for (int i = 0; i < size; i++) {
        maxVal = fmaxf(maxVal, z[i]);
    }

    // Step 3: Compute sum(exp(z_i - max))
    float sumExp = 0.0f;
    for (int i = 0; i < size; i++) {
        sumExp += expf(z[i] - maxVal);
    }

    __syncthreads();

    // Step 4: Compute softmax output
    if (idx < size) {
        neuronValues[idx] = expf(z[idx] - maxVal) / sumExp;
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
        int weightIndex = idx * previousLayerSize;

        float grad;
        if (activationFunction == 2) {
            //Crossentropy + sigmoid or softmax
			grad = learningRate * errors[idx];
        }
        else {
			grad = learningRate * errors[idx] * ActivationFunctions::ActivationDeriv(neuronValues[idx], activationFunction);
        }

        for (int j = 0; j < previousLayerSize; j++) {
            atomicAdd(&weights[weightIndex + j], learningRate * grad * prevNeuronValues[j]);
        }
        atomicAdd(&biases[idx], learningRate * grad);
    }
}

void OutputLayer::FeedForward(int threadsPerBlock) {
    //Compute neuron values for output layer

    //softmax index should be 2,
    //maybe I should use an enum on cuda too :D
	if (this->Activation == 2) {
        int blocks = 1;
        int threads = this->Size;
        size_t sharedMem = this->Size * sizeof(float);
        ff_outputValues_Softmax_GPU << <blocks, threads, sharedMem >> > (
            this->NeuronValues,
            this->previousLayer->NeuronValues,
            this->Weights,
            this->Biases,
            this->previousLayer->Size,
            this->Size
            );
    }

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