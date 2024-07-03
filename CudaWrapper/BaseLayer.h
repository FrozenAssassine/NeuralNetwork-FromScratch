#pragma once
class BaseLayer
{
public:
	BaseLayer* previousLayer;
	BaseLayer* nextLayer;
	float* Weights;
	float* Biases;
	float* Errors;
	float* NeuronValues;
	int Size;
	int Activation;

	virtual ~BaseLayer() = default;

	virtual void FeedForward(int threadsPerBlock) = 0;

	virtual void Train(int threadsPerBlock, float* desiredValues, float learningRate) = 0;
};
