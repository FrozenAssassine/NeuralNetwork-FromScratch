using System.Numerics;


namespace NNFromScratch.Optimizer;

internal class SIMDAccelerator
{
    public static readonly int SIMDLength = Vector<float>.Count;

    public LargeArray<float> Add(LargeArray<float> values1, LargeArray<float> values2)
    {
        var simdLength = SIMDLength;
        var result = new float[values1.Length];
        var i = 0;
        for (i = 0; i <= values1.Length - simdLength; i += simdLength)
        {
            var va = new Vector<float>(values1.Get(i, simdLength), i);
            var vb = new Vector<float>(values2.Get(i, simdLength), i);
            (va + vb).CopyTo(result, i);
        }

        for (; i < values1.Length; ++i)
        {
            result[i] = values1[i] + values2[i];
        }

        return (LargeArray<float>)result;
    }

    public float DotProduct(LargeArray<float> vec1, LargeArray<float> vec2)
    {
        var simdLength = SIMDLength;
        var result = new float[vec1.Length];
        var i = 0;
        for (i = 0; i <= vec1.Length - simdLength; i += simdLength)
        {
            var va = new Vector<float>(vec1.Get(i, simdLength), i);
            var vb = new Vector<float>(vec2.Get(i, simdLength), i);
            (va * vb).CopyTo(result, i);
        }

        for (; i < vec1.Length; ++i)
        {
            result[i] = vec1[i] * vec2[i];
        }

        return Sum(result);
    }

    public float DotProduct(LargeArray<float> vec1, LargeArray<LargeArray<float>> vec2, int subArrayIndex)
    {
        // difficult to accelerate in this form
        float res = 0;
        for (int i = 0; i < vec1.Length; i++)
        {
            res += vec1[i] * vec2[i][subArrayIndex];
        }
        return res;
    }

    public LargeArray<float> Multiply(LargeArray<float> values1, LargeArray<float> values2)
    {
        var simdLength = SIMDLength;
        var result = new float[values1.Length];
        var i = 0;
        for (i = 0; i <= values1.Length - simdLength; i += simdLength)
        {
            var va = new Vector<float>(values1.Get(i, simdLength), i);
            var vb = new Vector<float>(values2.Get(i, simdLength), i);
            (va * vb).CopyTo(result, i);
        }

        for (; i < values1.Length; ++i)
        {
            result[i] = values1[i] * values2[i];
        }

        return (LargeArray<float>)result;
    }

    public LargeArray<float> Multiply(float value1, LargeArray<float> values2)
    {
        LargeArray<float> res = new LargeArray<float>(values2.Length);
        for (int i = 0; i < values2.Length; i++)
            res[i] = value1 * values2[i];
        return res;
    }
    public float Sum(float[] nums, int start = 0)
    {
        var simdLength = SIMDLength;
        var simdPlusStart = simdLength + start;
        if (nums.Length - start < simdLength * 2)
        {
            float res = nums[start];
            for (int i = start + 1; i < nums.Length; i++)
                res += nums[i];
            return res;
        }
        else
        {
            var va = new Vector<float>(nums, start);
            var vb = new Vector<float>(nums, simdPlusStart);
            (va + vb).CopyTo(nums, simdPlusStart);
            return Sum(nums, simdPlusStart);
        }
    }
}
