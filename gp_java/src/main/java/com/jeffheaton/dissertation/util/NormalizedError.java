package com.jeffheaton.dissertation.util;

import org.encog.ml.MLRegression;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;

/**
 * Created by jeff on 7/11/16.
 */
public class NormalizedError {

    private double min;
    private double max;
    private double mean;
    private double sd;
    private int outputCount;

    public NormalizedError(MLDataSet theData) {
        this.min = Double.POSITIVE_INFINITY;
        this.max = Double.NEGATIVE_INFINITY;
        this.outputCount = 0;

        double sum = 0;
        for(MLDataPair pair: theData) {
            for(double d: pair.getIdealArray()) {
                this.min = Math.min(d,this.min);
                this.max = Math.max(d,this.max);
                sum += d;
                outputCount++;
            }
        }

        this.mean = sum / outputCount;

        for(MLDataPair pair: theData) {
            for(double d: pair.getIdealArray()) {
                double z = d - this.mean;
                sum += z * z;
            }
        }

        this.sd = Math.sqrt(sum/this.outputCount);
    }

    public double calculateNormalizedRange(MLDataSet theData,MLRegression theModel) {
        double sum = 0;
        for(MLDataPair pair: theData) {
            MLData actual = theModel.compute(pair.getInput());
            for(int i=0;i<pair.getIdeal().size();i++) {
                double d = actual.getData(i) - pair.getIdeal().getData(i);
                d = d * d;
                sum+=d;
            }
        }

        return Math.sqrt (sum/this.outputCount) / (this.max - this.min);
    }
}
