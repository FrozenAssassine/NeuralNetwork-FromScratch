#pragma once

#include <cmath>
#include <iostream>

class AccuracyCalculator
{

private:
    int totalCounter = 0;
    int correctCounter = 0;

    int GetMaximumIndex(float* arr, int n) {
        float max = arr[0];
        int index = 0;
        for (int i = 1; i < n; i++) {
            if (arr[i] > max) {
                max = arr[i];
                index = i;
            }
        }
        return index;
    }

public:

    void Calculate(float* inputX, float* desired, void (*predictFunction)(float*, float*), int samples, int features, int outputs, int startIndex = 0)
    {
        for (int i = startIndex; i < samples; i++)
        {
            float* x = &inputX[i * features];
            float* y = &desired[i * outputs];

            float* prediction = new float[outputs];
            predictFunction(x, prediction);

            int index1 = GetMaximumIndex(prediction, outputs);
            int index2 = GetMaximumIndex(y, outputs);
            
            if (index1 == index2)
                correctCounter++;
            totalCounter++;
        }
    }

    void PrintAccuracy()
    {
        printf("Accuracy %.5f%% (%d / %d)\n", correctCounter, totalCounter, MakeAccuracy());
    }
    float MakeAccuracy()
    {
        return ((float)correctCounter / totalCounter) * 100.0f;
    }
    void NextEpoch()
    {
        correctCounter = totalCounter = 0;
    }
};
