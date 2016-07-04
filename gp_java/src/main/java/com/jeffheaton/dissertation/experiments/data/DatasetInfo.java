package com.jeffheaton.dissertation.experiments.data;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Created by jeff on 7/4/16.
 */
public class DatasetInfo {

    private final boolean regression;
    private final String name;
    private final String target;
    private final List<String> predictors = new ArrayList<>();
    private final List<String> experiments = new ArrayList<>();
    private final int targetElements;

    public DatasetInfo(boolean theRegression, String theName, String theTarget,
                       List<String> thePredictors, List<String> theExperiments, int theTargetElements) {
        this.regression = theRegression;
        this.name = theName;
        this.target = theTarget;
        this.predictors.addAll(thePredictors);
        this.experiments.addAll(theExperiments);
        this.targetElements = theTargetElements;
    }

    public boolean isRegression() {
        return regression;
    }

    public String getName() {
        return name;
    }

    public String getTarget() {
        return target;
    }

    public List<String> getPredictors() {
        return predictors;
    }

    public List<String> getExperiments() {
        return experiments;
    }

    public int getTargetElements() {
        return targetElements;
    }

    public String toString() {
        StringBuilder result = new StringBuilder();
        result.append("[Dataset:");
        result.append(this.name);
        result.append(",regression=");
        result.append(this.regression);
        result.append(",target=");
        result.append(this.target);
        result.append(",predictors=");
        result.append(this.predictors);
        result.append(",experiments=");
        result.append(this.experiments);
        result.append(",targetElements=");
        result.append(this.targetElements);
        result.append("]");
        return result.toString();
    }
}
