#pragma once
#include "BaseLayer.h"

class DenseLayer : public BaseLayer
{
public:

	virtual void FeedForward(int threadsPerBlock);

	virtual void Train(int threadsPerBlock, float* desiredValues, float learningRate);
};
