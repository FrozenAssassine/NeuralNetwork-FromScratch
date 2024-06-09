using Microsoft.VisualBasic;
using NNFromScratch.Core;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNFromScratch.Helper;

internal class SVGDrawingHelper
{
    private int distancePerLayer = 300;
    private int neuronRadius = 50;
    private int yDistPerNeuron = 50;
    private int shiftX = 100;
    private int shiftY = 100;

    public string DrawNeuron(int x, int y, Layer layer, int neuronIndex)
    {
        return $"<circle cx=\"{x}\" cy=\"{y}\" r=\"{neuronRadius}\" stroke=\"blue\" stroke-width=\"5\" fill=\"green\" />" +
            $"<text x=\"{x}\" y=\"{y}\" text-anchor=\"middle\" alignment-baseline=\"middle\" font-size=\"20\" fill=\"black\">{layer.Biases[neuronIndex]}</text>"; //{Math.Round(neuron.bias, 5)}
    }

    public string DrawLayer(int layerIndex, int maxHeight, Layer layer)
    {
        int layerHeight = CalculateMaxHeight(layer);
        int yStart = shiftY + (maxHeight - layerHeight) / 2;
        int xPos = shiftX + (layerIndex * distancePerLayer);

        StringBuilder layerSVG = new StringBuilder();
        for (int i = 0; i < layer.Size; i++)
        {
            layerSVG.AppendLine(DrawNeuron(xPos, yStart + (i * (yDistPerNeuron + neuronRadius * 2)), layer, i));
        }
        return layerSVG.ToString();
    }

    public int CalculateMaxHeight(NeuralNetwork nn)
    {
        List<Layer> all = new List<Layer>(nn.hiddenLayers);
        all.Add(nn.outputLayer);
        all.Add(nn.inputLayer);

        int max = all.Max(x => x.Size);
        return max * (neuronRadius * 2 + yDistPerNeuron) + shiftY;
    }
    public int CalculateMaxHeight(Layer layer)
    {
        return layer.Size * (neuronRadius * 2 + yDistPerNeuron) + shiftY;
    }

    public void DrawConnections(NeuralNetwork nn)
    {
        for(int i = 0; i<)

        //var str = $"<line x1=\"{x1}\" y1=\"{y1}\" x2=\"{x2}\" y2=\"{y2}\" stroke=\"black\" stroke-width=\"2\"/>\r\n";
        
        //layer.

        //return str;
    }

    public void Draw(NeuralNetwork network)
    {
        var height = CalculateMaxHeight(network);

        StringBuilder svg = new StringBuilder();
        svg.AppendLine($"<?xml version=\"1.0\" encoding=\"UTF-8\"?>\r\n<svg width=\"1000\" height=\"{height}\" xmlns=\"http://www.w3.org/2000/svg\">");
        svg.AppendLine(DrawLayer(0, height, network.inputLayer));
        
        for(int i = 0; i < network.hiddenLayers.Length; i++)
        {
            svg.AppendLine(DrawLayer(i + 1, height, network.hiddenLayers[i]));
        }

        svg.AppendLine(DrawLayer(network.hiddenLayers.Length + 1, height, network.outputLayer));

        svg.AppendLine("</svg>");
        File.WriteAllText("D:\\nn.svg", svg.ToString());
    }
}
