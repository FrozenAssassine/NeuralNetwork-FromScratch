#pragma once
#include "DenseLayer.h"

class OutputLayer : public DenseLayer
{
public:
	void OutputLayer::FeedForward(int threadsPerBlock) override;
	void OutputLayer::Train(int threadsPerBlock, float* desiredValues, float learningRate) override;
};