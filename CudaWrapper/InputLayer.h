#pragma once
#include "DenseLayer.h"

class InputLayer : public DenseLayer
{
public:
	void FeedForward(int threadsPerBlock) override;
	void Train(int threadsPerBlock, float* desiredValues, float learningRate) override;
};