package com.jeffheaton.dissertation.util;

import org.encog.EncogError;
import org.encog.ml.MLRegression;
import org.encog.ml.data.MLDataSet;
import org.encog.neural.networks.BasicNetwork;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by jeff on 6/17/16.
 */
public class PermutationFeatureImportanceCalc {
    private final List<FeatureRanking> features = new ArrayList<>();
    private final MLRegression model;
    private final MLDataSet dataset;

    public PermutationFeatureImportanceCalc(MLRegression theModel, MLDataSet theDataset) {
        this.model = theModel;
        this.dataset = theDataset;

        for (int i = 0; i < theModel.getInputCount(); i++) {
            this.features.add(new FeatureRanking("f" + i));
        }
    }

    public PermutationFeatureImportanceCalc(MLRegression theModel, MLDataSet theDataset,
                                            String[] theFeatureNames) {
        this.model = theModel;
        this.dataset = theDataset;

        if (theFeatureNames.length != this.model.getInputCount()) {
            throw new EncogError("Neural network inputs(" + this.model.getInputCount() + ") and feature name count("
                    + theFeatureNames.length + ") do not match.");
        }

        for (String name : theFeatureNames) {
            this.features.add(new FeatureRanking(name));
        }
    }

    public void reset() {
        for (FeatureRanking rank : this.features) {
            rank.setImportancePercent(0);
            rank.setTotalWeight(0);
        }
    }

    @Override
    public String toString() {
        return this.features.toString();
    }

    public List<FeatureRanking> getFeatures() {
        return features;
    }

    public void calculateFeatureImportance() {

    }

}
