#pragma once
#include "BaseLayer.h"

class LSTMLayer : public BaseLayer
{
public:
    float* CellState;
    float* OutputGate;
    float* ForgetGate;
    float* InputGate;
    float* CandidateCellState;

    float* WeightsInput;
    float* WeightsForget;
    float* WeightsOutput;
    float* WeightsCandidate;

    float* outputGradients;
    float* cellStateGradients;
    float* inputGateGradients;
    float* forgetGateGradients;
    float* candidateCellGradients;

    void FeedForward(int threadsPerBlock) override;
    void Train(int threadsPerBlock, float* desiredValues, float learningRate) override;
};

