using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNFromScratch.Helper
{
    internal static class ArrayHelper
    {
        public static IEnumerable<float> GetItems(this float[] array, int start, int count)
        {
            for(int i = 0;  i < count; i++) 
            {
                yield return array[start + i];
            }
        }
    }
}
