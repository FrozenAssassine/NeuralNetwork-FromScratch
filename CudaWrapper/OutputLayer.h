#pragma once
#include "BaseLayer.h"

class OutputLayer : public BaseLayer
{
public:
	void OutputLayer::FeedForward(int threadsPerBlock) override;
	void OutputLayer::Train(int threadsPerBlock, float* desiredValues, float learningRate) override;
};