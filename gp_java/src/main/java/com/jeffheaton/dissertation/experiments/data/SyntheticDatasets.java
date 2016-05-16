package com.jeffheaton.dissertation.experiments.data;

import org.encog.mathutil.EncogFunction;
import org.encog.mathutil.randomize.generate.GenerateRandom;
import org.encog.mathutil.randomize.generate.MersenneTwisterGenerateRandom;
import org.encog.ml.data.MLDataSet;

/**
 * Created by jeff on 5/10/16.
 */
public class SyntheticDatasets {
    public static MLDataSet generatePolynomial() {
        GenerateRandom rnd = new MersenneTwisterGenerateRandom(42);
        return SampleFromFunction.sample(rnd,10000,0,2,
                new EncogFunction() {

                    @Override
                    public double fn(double[] x) {
                        // return (x[0] + 10) / 4;
                        // return Math.sin(x[0]);
                        return 1 + (5 * x[0]) + (8 * Math.pow(x[0], 2));
                    }

                    @Override
                    public int size() {
                        return 1;
                    }

                });
    }

    public static MLDataSet generateDiffRatio() {
        GenerateRandom rnd = new MersenneTwisterGenerateRandom(42);
        return SampleFromFunction.sample(rnd,10000,1,10,
                new EncogFunction() {

                    @Override
                    public double fn(double[] x) {
                        // return (x[0] + 10) / 4;
                        // return Math.sin(x[0]);
                        return (x[0]-x[1])/(x[2]-x[3]);
                    }

                    @Override
                    public int size() {
                        return 4;
                    }

                });
    }
}
