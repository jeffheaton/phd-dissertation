package com.jeffheaton.dissertation.experiments;

import com.jeffheaton.dissertation.data.AutoMPG;
import com.jeffheaton.dissertation.features.AutoEngineerFeatures;
import com.jeffheaton.dissertation.util.Transform;
import org.encog.mathutil.error.ErrorCalculation;
import org.encog.mathutil.error.ErrorCalculationMode;
import org.encog.mathutil.randomize.generate.MersenneTwisterGenerateRandom;
import org.encog.ml.data.MLDataSet;
import org.encog.util.csv.CSVFormat;
import org.encog.util.csv.ReadCSV;

import java.io.InputStream;

/**
 * Created by Jeff on 4/2/2016.
 */
public class ExperimentAutoFeature {

    public static void main(String[] args) {
        ErrorCalculation.setMode(ErrorCalculationMode.RMS);

        MLDataSet dataset = AutoMPG.getInstance().loadData();
        Transform.zscore(dataset);

        // split
        MLDataSet[] split = Transform.splitTrainValidate(dataset,new MersenneTwisterGenerateRandom(42),0.75);
        MLDataSet trainingSet = split[0];
        MLDataSet validationSet = split[1];


        AutoEngineerFeatures auto = new AutoEngineerFeatures(trainingSet, validationSet);
        auto.run();
    }
}
