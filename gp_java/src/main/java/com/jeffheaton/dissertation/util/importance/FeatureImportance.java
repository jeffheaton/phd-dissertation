package com.jeffheaton.dissertation.util.importance;

import org.encog.ml.MLInputOutput;
import org.encog.ml.MLMethod;
import org.encog.ml.MLRegression;
import org.encog.ml.data.MLDataSet;

import java.util.List;
import java.util.Set;

/**
 * Created by jeffh on 6/29/2016.
 */
public interface FeatureImportance {
    void init(MLRegression theModel, String[] names);
    void performRanking();
    void performRanking(MLDataSet theDataset);
    List<FeatureRank> getFeatures();
    MLRegression getModel();
}
