package com.jeffheaton.dissertation.util;

import org.encog.EncogError;
import org.encog.mathutil.randomize.generate.GenerateRandom;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;

/**
 * Created by Jeff on 3/31/2016.
 */
public class Transform {

    public static MLDataSet[] splitTrainValidate(MLDataSet trainingSet, GenerateRandom rnd, double trainingPercent) {
        if( trainingPercent<0 || trainingPercent>1) {
            throw new EncogError("Training percent must be between 0 and 1.");
        }

        MLDataSet[] result = new MLDataSet[2];
        result[0] = new BasicMLDataSet();
        result[1] = new BasicMLDataSet();

        // initial split
        for(MLDataPair pair: trainingSet ) {
            if( rnd.nextDouble()<trainingPercent ) {
                result[0].add(pair);
            } else {
                result[1].add(pair);
            }
        }

        return result;
    }

    public static void interpolate(MLDataSet data) {
        double[] sumInput = new double[data.getInputSize()];
        double[] sumIdeal = new double[data.getIdealSize()];

        // Pass 1: find means
        for(MLDataPair pair: data) {
            // Input
            for(int i=0;i<sumInput.length;i++) {
                double d = pair.getInput().getData(i);
                if(!Double.isNaN(d)) {
                    sumInput[i]+=d;
                }
            }

            // Ideal
            for(int i=0;i<sumIdeal.length;i++) {
                double d = pair.getIdeal().getData(i);
                if(!Double.isNaN(d)) {
                    sumIdeal[i]+=d;
                }
            }
        }
        // calculate means
        for(int i=0;i<sumInput.length;i++) {
            sumInput[i]/=data.size();
        }

        for(int i=0;i<sumIdeal.length;i++) {
            sumIdeal[i]/=data.size();
        }

        // pass 2:
        for(MLDataPair pair: data) {
            // Input
            for(int i=0;i<sumInput.length;i++) {
                double d = pair.getInput().getData(i);
                if(Double.isNaN(d)) {
                    pair.getInput().setData(i,sumInput[i]);
                }
            }

            // Ideal
            for(int i=0;i<sumIdeal.length;i++) {
                double d = pair.getIdeal().getData(i);
                if(Double.isNaN(d)) {
                    pair.getIdeal().setData(i,sumIdeal[i]);
                }
            }
        }
    }


    public static void zscore(MLDataSet dataset) {
        double[] mean = new double[dataset.getInputSize()];

        // Pass 1: find means
        for(MLDataPair pair: dataset) {
            // Input
            for(int i=0;i<mean.length;i++) {
                double d = pair.getInput().getData(i);
                mean[i]+=d;
            }
        }

        // calculate means
        for(int i=0;i<mean.length;i++) {
            mean[i]/=dataset.size();
        }

        // pass 2: standard dev
        double[] sdev = new double[dataset.getInputSize()];

        for(MLDataPair pair: dataset) {
            // Input
            for(int i=0;i<mean.length;i++) {
                double d = pair.getInput().getData(i)-mean[i];
                sdev[i] += d * d;
            }
        }

        for(int i=0;i<sdev.length;i++) {
            sdev[i] = Math.sqrt(sdev[i]/dataset.size());
        }

        // pass 3: zscore
        for(MLDataPair pair: dataset) {
            // Input
            for(int i=0;i<mean.length;i++) {
                double d = pair.getInput().getData(i)-mean[i];
                pair.getInput().setData(i,d/sdev[i]);
            }
        }

    }
}
