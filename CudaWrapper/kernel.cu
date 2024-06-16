#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_runtime.h>

#include <chrono>
#include <stdio.h>
#include <iostream>
#include <vector>

int threadsPerBlock = 1024;

typedef struct Layer {
    float* Biases;
    float* NeuronValues;
    float* Errors;
    float* Weights;
    int Size;
} Layer;

Layer* gpuLayers;
Layer* cpuLayers;

float* desiredValues;

int allLayersCount;

#define CUDA_CHECK(err, code) \
    if (err != cudaSuccess) { \
        printf("\nCUDA error (%s): %s at line %d\n", code, cudaGetErrorString(err), __LINE__); \
    }

__global__ void feedForwardKernel(float* prevNeuronValues, float* weights, float* biases, float* neuronValues, int prevSize, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float sum = 0.0f;
        for (int j = 0; j < prevSize; j++) {
            sum += prevNeuronValues[j] * weights[idx * prevSize + j];
        }
        neuronValues[idx] = 1.0f / (1.0f + expf(-(sum + biases[idx]))); // Sigmoid activation
    }
}

__global__ void feedForwardKernelOutputLayer(int outLayerPrevSize, float* output_weights, float* output_biases, float* output_neuronValues,
    float* outputPrev_neuronValues, int prevSize, int outLayerSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < outLayerSize) {
        float sum = 0.0f;
        for (int j = 0; j < prevSize; j++)
        {
            int weightIndex = idx * outLayerPrevSize + j;
            sum += outputPrev_neuronValues[j] * output_weights[weightIndex];
        }
        output_neuronValues[idx] = 1.0f / (1.0f + expf(-(sum + output_biases[idx])));
    }
}

__global__ void calculateOutputErrors(float* desiredOutputs, float* ffoutputValues, float* outLayerErrors, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        outLayerErrors[idx] = desiredOutputs[idx] - ffoutputValues[idx];
    }
}

__global__ void updateWeightsAndBiases(float* currentLayerErrors, float* currentLayerNeuronValues, float* prevNeuronValues,
    float* currentLayerWeights, float* currentLayerBiases, int prevSize, int size, float learningRate)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        for (int j = 0; j < prevSize; j++) {
            int weightIndex = idx * prevSize + j;
            currentLayerWeights[weightIndex] += learningRate * currentLayerErrors[idx] * prevNeuronValues[j];
        }
        currentLayerBiases[idx] += learningRate * currentLayerErrors[idx];
    }
}

__global__ void propagateErrorsKernel(float* nextLayerErrors, float* nextWeights, float* currentLayerErrors, float* currentLayerNeuronValues, int nextSize, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float error = 0.0f;
        for (int j = 0; j < nextSize; j++) {
            error += nextLayerErrors[j] * nextWeights[j * size + idx];
        }
        currentLayerErrors[idx] = error * (currentLayerNeuronValues[idx] * (1.0f - currentLayerNeuronValues[idx])); // Sigmoid derivative
    }
}

__global__ void updateWeightsAndBiasesOutputLayer(int size, float learningRate, float* outputLayerWeights, int prevLayerSize,
    float* outputLayerNeuronValues, float* prevOutputLayerNeuronValues, float* outputLayerError, float* outputLayerBiases)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        for (int j = 0; j < prevLayerSize; j++)
        {
            int weightIndex = idx * prevLayerSize + j;
            outputLayerWeights[weightIndex] += learningRate * outputLayerError[idx] * (outputLayerNeuronValues[idx] * (1 - outputLayerNeuronValues[idx])) * prevOutputLayerNeuronValues[j];
        }

        outputLayerBiases[idx] += learningRate * outputLayerError[idx] * (outputLayerNeuronValues[idx] * (1 - outputLayerNeuronValues[idx]));
    }
}

__global__ void test(float* weights, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        weights[idx] = 0.1f;
    }
}


float* FeedForward(float* data, int n, bool allocate = true) {
    cudaError_t err;

    //set this to true, if the data is not already on the gpu from the train function
    if (allocate) {
        err = cudaMemcpy(gpuLayers[0].NeuronValues, data, n * sizeof(float), cudaMemcpyHostToDevice);
        CUDA_CHECK(err, "0");
    }

    for (int i = 1; i < allLayersCount - 1; i++) {
        Layer& prevLayer = gpuLayers[i - 1];
        Layer& currLayer = gpuLayers[i];

        int blocks = (currLayer.Size + threadsPerBlock - 1) / threadsPerBlock;

        feedForwardKernel << <blocks, threadsPerBlock >> > (
            prevLayer.NeuronValues,
            currLayer.Weights,
            currLayer.Biases,
            currLayer.NeuronValues,
            prevLayer.Size,
            currLayer.Size
            );

        cudaDeviceSynchronize();
        err = cudaGetLastError();
        CUDA_CHECK(err, "1");
    }

    Layer& prevLayer = gpuLayers[allLayersCount - 2];
    Layer& outLayer = gpuLayers[allLayersCount - 1];

    int blocks = (outLayer.Size + threadsPerBlock - 1) / threadsPerBlock;
    feedForwardKernelOutputLayer << <blocks, threadsPerBlock >> > (
        prevLayer.Size,
        outLayer.Weights,
        outLayer.Biases,
        outLayer.NeuronValues,
        prevLayer.NeuronValues,
        prevLayer.Size,
        outLayer.Size
        );

    cudaDeviceSynchronize();
    err = cudaGetLastError();
    CUDA_CHECK(err, "2");

    //printf("Done FeedForward\n");
    return outLayer.NeuronValues;
}

extern "C" __declspec(dllexport) void Predict(float* data, float* prediction, int n) {
    float* res = FeedForward(data, n, true);

    cudaMemcpy(prediction, data, n * sizeof(float), cudaMemcpyDeviceToHost);
}

extern "C" __declspec(dllexport) void Train(float* inputs, float* desiredOutputs, int size, float learningRate) {
    cudaError_t err;
    Layer& outputLayer = gpuLayers[allLayersCount - 1];
    Layer& prevLayer = gpuLayers[allLayersCount - 2];

    //copy the new inputs and outputs to the gpu
    err = cudaMemcpy(gpuLayers[0].NeuronValues, inputs, cpuLayers[0].Size * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK(err, "3");

    if (desiredValues == nullptr) {
        cudaMalloc(&desiredValues, cpuLayers[allLayersCount - 1].Size * sizeof(float));
    }

    err = cudaMemcpy(desiredValues, desiredOutputs, cpuLayers[allLayersCount - 1].Size * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK(err, "4");

    // Perform feedforward pass to get the network's output
    FeedForward(gpuLayers[0].NeuronValues, size, false);

    // Calculate errors for the output layer
    int outputBlocks = (outputLayer.Size + threadsPerBlock - 1) / threadsPerBlock;

    calculateOutputErrors << <outputBlocks, threadsPerBlock >> > (
        gpuLayers[allLayersCount - 1].NeuronValues,
        desiredValues,
        outputLayer.Errors,
        outputLayer.Size
        );
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    CUDA_CHECK(err, "5");

    // Update weights and biases for the output layer
    updateWeightsAndBiasesOutputLayer << <outputBlocks, threadsPerBlock >> > (
        outputLayer.Size,
        learningRate,
        outputLayer.Weights,
        prevLayer.Size,
        outputLayer.NeuronValues,
        prevLayer.NeuronValues,
        outputLayer.Errors,
        outputLayer.Biases
        );
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    CUDA_CHECK(err, "6");

    // Backpropagate the errors to the hidden layers
    for (int i = allLayersCount - 2; i >= 1; i--) {
        Layer& currLayer = gpuLayers[i];
        Layer& prevLayer = gpuLayers[i - 1];
        Layer& nextLayer = gpuLayers[i + 1];

        //int errorBlocks = (nextLayer.Size + threadsPerBlock - 1) / threadsPerBlock;
        int errorBlocks = (currLayer.Size + threadsPerBlock - 1) / threadsPerBlock;
        propagateErrorsKernel << <errorBlocks, threadsPerBlock >> > (
            nextLayer.Errors,
            nextLayer.Weights,
            currLayer.Errors,
            currLayer.NeuronValues,
            nextLayer.Size,
            currLayer.Size
            );
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        CUDA_CHECK(err, "7");

        int updateBlocks = (currLayer.Size + threadsPerBlock - 1) / threadsPerBlock;
        updateWeightsAndBiases << <updateBlocks, threadsPerBlock >> > (
            currLayer.Errors,
            currLayer.NeuronValues,
            prevLayer.NeuronValues,
            currLayer.Weights,
            currLayer.Biases,
            prevLayer.Size,
            currLayer.Size,
            learningRate
            );

        cudaDeviceSynchronize();
        err = cudaGetLastError();
        CUDA_CHECK(err, "8");
    }
}


extern "C" __declspec(dllexport) void Cleanup() {
    //free the memory of every layer from the gpu
    for (int i = 0; i < allLayersCount; i++) {
        cudaFree(gpuLayers[i].Biases);
        cudaFree(gpuLayers[i].NeuronValues);
        cudaFree(gpuLayers[i].Errors);
        cudaFree(gpuLayers[i].Weights);
    }
    cudaFree(desiredValues);
    delete[] gpuLayers;
}

extern "C" __declspec(dllexport) void DoneTraining() {
    cudaError_t err;

    // Copy updated weights and biases back to the host
    for (int i = 0; i < allLayersCount; i++) {
        Layer& gpuLayer = gpuLayers[i];
        Layer& cpuLayer = cpuLayers[i];

        //copy back all layers:
        err = cudaMemcpy(cpuLayer.Biases, gpuLayer.Biases, gpuLayer.Size * sizeof(float), cudaMemcpyDeviceToHost);
        CUDA_CHECK(err, "10");
        err = cudaMemcpy(cpuLayer.Errors, gpuLayer.NeuronValues, gpuLayer.Size * sizeof(float), cudaMemcpyDeviceToHost);
        CUDA_CHECK(err, "11");
        err = cudaMemcpy(cpuLayer.NeuronValues, gpuLayer.NeuronValues, gpuLayer.Size * sizeof(float), cudaMemcpyDeviceToHost);
        CUDA_CHECK(err, "12");

        //input layer doesn't have any weights:
        if (i != 0) {
            err = cudaMemcpy(cpuLayer.Weights, gpuLayer.Weights, gpuLayer.Size * gpuLayers[i - 1].Size * sizeof(float), cudaMemcpyDeviceToHost);
            CUDA_CHECK(err, "9");
            printf("SIZE: (%d), %d\n", i, (gpuLayer.Size * gpuLayers[i - 1].Size));
            //printf("Copy weights");
        }
        printf("MemoryAdress: %d, %p, %p, %p, %p\n", i, cpuLayer.Weights, cpuLayer.Biases, cpuLayer.Errors, cpuLayer.NeuronValues);
    }
    Cleanup();
}

extern "C" __declspec(dllexport) void Init(int totalLayer) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    threadsPerBlock = prop.maxThreadsPerBlock;

    allLayersCount = totalLayer;

    gpuLayers = new Layer[totalLayer];
    cpuLayers = new Layer[totalLayer];
}

extern "C" __declspec(dllexport) void Test() {

    //only to see whether the update of the weights works
    int updateBlocks = ((gpuLayers[1].Size * gpuLayers[0].Size) + threadsPerBlock - 1) / threadsPerBlock;
    test << <updateBlocks, threadsPerBlock >> > (gpuLayers[1].Weights, gpuLayers[1].Size * gpuLayers[0].Size);
}

extern "C" __declspec(dllexport) void InitLayer(int layerIndex, int prevSize, int size, float* biases, float* weights, float* values, float* errors) {
    printf("Initializing layer %d with size %d and prevSize %d\n", layerIndex, size, prevSize);

    Layer& gpuLayer = gpuLayers[layerIndex];
    gpuLayer.Size = size;

    Layer& cpuLayer = cpuLayers[layerIndex];
    cpuLayer.Size = size;
    cpuLayer.Biases = biases;
    cpuLayer.Errors = errors;
    cpuLayer.NeuronValues = values;
    cpuLayer.Weights = weights;

    // Print the host pointers to verify they are valid
    printf("Host Biases: %p\n", biases);
    printf("Host Weights: %p\n", weights);
    printf("Host NeuronValues: %p\n", values);
    printf("Host Errors: %p\n", errors);

    // Allocate GPU memory
    cudaError_t err;
    err = cudaMalloc(&gpuLayer.Biases, size * sizeof(float));
    CUDA_CHECK(err, "25");
    err = cudaMalloc(&gpuLayer.NeuronValues, size * sizeof(float));
    CUDA_CHECK(err, "26");
    err = cudaMalloc(&gpuLayer.Errors, size * sizeof(float));
    CUDA_CHECK(err, "27");

    if (prevSize != 0) {
        err = cudaMalloc(&gpuLayer.Weights, size * prevSize * sizeof(float));
        CUDA_CHECK(err, "28");
    }

    // Print the GPU pointers to ensure they are allocated
    printf("GPU Biases: %p\n", gpuLayer.Biases);
    printf("GPU NeuronValues: %p\n", gpuLayer.NeuronValues);
    printf("GPU Errors: %p\n", gpuLayer.Errors);
    printf("GPU Weights: %p\n", gpuLayer.Weights);

    // Copy initial data to GPU
    err = cudaMemcpy(gpuLayer.Biases, cpuLayer.Biases, size * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK(err, "29");
    err = cudaMemcpy(gpuLayer.NeuronValues, cpuLayer.NeuronValues, size * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK(err, "30");
    err = cudaMemcpy(gpuLayer.Errors, cpuLayer.Errors, size * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK(err, "31");
    if (prevSize != 0) {
        err = cudaMemcpy(gpuLayer.Weights, cpuLayer.Weights, size * prevSize * sizeof(float), cudaMemcpyHostToDevice);
        CUDA_CHECK(err, "32");
        printf("SIZE: (%d), %d\n", layerIndex, (size * prevSize));
    }

    printf("Layer %d initialized successfully\n", layerIndex);
}
