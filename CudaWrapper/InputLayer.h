#pragma once
#include "BaseLayer.h"

class InputLayer : public BaseLayer
{
public:
	void FeedForward(int threadsPerBlock) override;
	void Train(int threadsPerBlock, float* desiredValues, float learningRate) override;
};