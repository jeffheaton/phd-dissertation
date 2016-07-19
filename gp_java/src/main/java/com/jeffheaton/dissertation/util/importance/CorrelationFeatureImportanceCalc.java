package com.jeffheaton.dissertation.util.importance;

import org.encog.EncogError;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;

/**
 * Compute feature importance using correlation between each of the features and the target.  This ranker should only
 * be used with datasets that are either regression or binary classifiers.  If there are multiple outputs, this ranker
 * will throw an error.  Also, if there are are more than 2 classes encoded into a single output neuron (e.g. a
 * SVM designed for more than two classes), this ranker should not be used.
 *
 * This ranker does not require a trained model, it simply looks at the training data and determines how closely each
 * of the inputs correlates to the target.
 *
 * https://en.wikipedia.org/wiki/Correlation_and_dependence
 */
public class CorrelationFeatureImportanceCalc extends AbstractFeatureImportance {

    /**
     * {@inheritDoc}
     */
    @Override
    public void performRanking() {
        throw new EncogError("This algorithm requires a dataset to measure performance against, please call performRanking with a dataset.");
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void performRanking(MLDataSet theDataset) {
        if( getModel() !=null ) {
            throw new EncogError("This algorithm does not use a model, please pass null to init for the model.");
        }

        if( theDataset.getIdealSize() !=1 ) {
            throw new EncogError("This algorithm can only be used with a single-output dataset.");
        }

        // Calculate sums for the mean values
        double[] mean = new double[1+theDataset.getInputSize()];
        for(MLDataPair pair: theDataset) {
            mean[0] += pair.getIdeal().getData(0);
            for(int i=0;i<pair.getInput().size();i++) {
                mean[1+i] += pair.getInput().getData(i);
            }
        }

        // Calculate the means
        for(int i=0;i<mean.length;i++) {
            mean[i]/=theDataset.size();
        }

        // Calculate the variance
        double[] sd = new double[1+theDataset.getInputSize()];
        for(MLDataPair pair: theDataset) {
            double d = pair.getIdeal().getData(0) - mean[0];
            sd[0] += d*d;
            for(int i=0;i<pair.getInput().size();i++) {
                d = pair.getInput().getData(i) - mean[1+i];
                sd[1+i] += d*d;
            }
        }

        // Turn variance to SD
        for(int i=0;i<mean.length;i++) {
            sd[i]=Math.sqrt(sd[i]/theDataset.size());
        }

        // Compute correlation
        for(int i=0;i<theDataset.getInputSize();i++) {
            FeatureRank rank = this.getFeatures().get(i);

            double acc = 0;
            for(MLDataPair pair: theDataset) {
                double a = pair.getIdeal().getData(0) - mean[0];
                double b = pair.getInput().getData(i) - mean[i+1];
                acc+=a*b;
            }

            double cov = (1.0/(1.0-theDataset.size()))*acc;
            double cor = cov / (sd[0]*sd[i+1]);
            rank.setImportancePercent(Math.abs(cor));
            rank.setTotalWeight(cor);
        }
    }
}
