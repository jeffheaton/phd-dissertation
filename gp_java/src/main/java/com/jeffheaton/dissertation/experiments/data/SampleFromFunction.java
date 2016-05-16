package com.jeffheaton.dissertation.experiments.data;

import org.encog.mathutil.EncogFunction;
import org.encog.mathutil.randomize.generate.GenerateRandom;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataSet;

/**
 * Created by jeff on 5/10/16.
 */
public class SampleFromFunction {
    public static MLDataSet sample(GenerateRandom rnd, int rowCount, double xMin, double xMax, EncogFunction fn) {
        BasicMLDataSet result = new BasicMLDataSet();
        double[] current = new double[fn.size()];

        for(int i=0;i <rowCount; i++) {
            for(int j=0;j<current.length;j++) {
                current[j] = rnd.nextDouble(xMin,xMax);
            }

            MLData input = new BasicMLData(current);
            MLData ideal = new BasicMLData(1);
            ideal.setData(0, fn.fn(current));
            result.add(input,ideal);
        }

        return result;
    }
}
