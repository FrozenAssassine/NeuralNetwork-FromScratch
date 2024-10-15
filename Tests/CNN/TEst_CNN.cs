using NNFromScratch.Core.Layers;
using NNFromScratch.Core;
using NNFromScratch.Models;

namespace Tests.CNN;

internal class Test_CNN
{
    int imageWidth = 0;
    int imageHeight = 0;

    public void Run()
    {

        //create the neural network:
        var network = NetworkBuilder.Create()
            .Stack(new InputLayer(imageWidth * imageHeight))
            .Stack(new ConvolutionalLayer(imageWidth, imageHeight, 1, new ConvolutionalFilterType[] { ConvolutionalFilterType.SobelX, ConvolutionalFilterType.SobelY } ))
            .Stack(new DenseLayer(256, ActivationType.Sigmoid))
            .Stack(new OutputLayer(10, ActivationType.Softmax))
            .Build(true);
    }
}
