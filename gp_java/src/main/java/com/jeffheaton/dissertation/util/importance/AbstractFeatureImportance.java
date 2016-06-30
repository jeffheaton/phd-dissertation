package com.jeffheaton.dissertation.util.importance;

import org.encog.EncogError;
import org.encog.ml.MLInputOutput;
import org.encog.ml.MLMethod;
import org.encog.ml.MLRegression;
import org.encog.ml.data.MLDataSet;

import java.util.*;

/**
 * Created by jeffh on 6/29/2016.
 */
public abstract class AbstractFeatureImportance implements FeatureImportance {

    private MLRegression model;
    private final List<FeatureRank> features = new ArrayList<>();

    @Override
    public void init(MLRegression theModel, String[] theFeatureNames) {
        this.model = theModel;

        if( theFeatureNames==null ) {
            for (int i = 0; i < this.model.getInputCount(); i++) {
                this.features.add(new FeatureRank("f" + i));
            }
        } else {
            if (theFeatureNames.length != this.model.getInputCount()) {
                throw new EncogError("Neural network inputs(" + this.model.getInputCount() + ") and feature name count("
                        + theFeatureNames.length + ") do not match.");
            }

            for (String name : theFeatureNames) {
                this.features.add(new FeatureRank(name));
            }
        }
    }

    @Override
    public List<FeatureRank> getFeatures() {
        return this.features;
    }

    public List<FeatureRank> getFeaturesSorted() {
        ArrayList<FeatureRank> result = new ArrayList<>();
        result.addAll(this.features);
        result.sort(new Comparator<FeatureRank>() {
            @Override
            public int compare(FeatureRank o1, FeatureRank o2) {
                return Double.compare(o2.getImportancePercent(), o1.getImportancePercent());
            }
        });
        return result;
    }

    @Override
    public String toString() {
        StringBuilder result = new StringBuilder();
        for (FeatureRank ranking : getFeaturesSorted()) {
            int idx = getFeatures().indexOf(ranking);
            if( result.length()>0) {
                result.append(",");
            }
            result.append(idx);
        }
        return result.toString();
    }

    @Override
    public MLRegression getModel() {
        return this.model;
    }
}
