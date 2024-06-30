
using NNFromScratch.Helper;

namespace NNFromScratch.Core.Layers
{
    internal class LSTMLayer
    {
        //forget gate
        private float wf;
        private float[] bf;

        //input gate
        private float wi;
        private float[] bi;

        //candidate gate
        public LSTMLayer(int size)
        {

            wf = MathHelper.RandomWeight();
            bf = new float[size];
            wi = MathHelper.RandomWeight();
            bi = new float[size];


        }
    }
}
