package com.jeffheaton.dissertation.experiments.misc;

import com.jeffheaton.dissertation.JeffDissertation;
import com.jeffheaton.dissertation.autofeatures.AutoEngineerFeatures;
import com.jeffheaton.dissertation.experiments.manager.DissertationConfig;
import com.jeffheaton.dissertation.util.QuickEncodeDataset;
import org.encog.mathutil.error.ErrorCalculation;
import org.encog.mathutil.error.ErrorCalculationMode;
import org.encog.mathutil.randomize.generate.MersenneTwisterGenerateRandom;
import org.encog.ml.data.MLDataSet;
import org.encog.persist.source.ObtainFallbackStream;
import org.encog.persist.source.ObtainInputStream;
import org.encog.persist.source.ObtainResourceInputStream;
import org.encog.util.csv.CSVFormat;
import org.encog.util.simple.EncogUtility;

import java.io.File;

public class ExperimentAutoFeature {

    public static void main(String[] args) {
        ErrorCalculation.setMode(ErrorCalculationMode.RMS);

        ObtainInputStream source = new ObtainFallbackStream(
                DissertationConfig.getInstance().getDataPath().toString(),
                "auto-mpg.csv", JeffDissertation.class);

        QuickEncodeDataset quick = new QuickEncodeDataset(false,false);
        quick.analyze(source,"mpg", true, CSVFormat.EG_FORMAT);
        MLDataSet dataset = quick.generateDataset();

        // split
        MLDataSet[] split = EncogUtility.splitTrainValidate(dataset,new MersenneTwisterGenerateRandom(42),0.75);
        MLDataSet trainingSet = split[0];
        MLDataSet validationSet = split[1];


        AutoEngineerFeatures auto = new AutoEngineerFeatures(trainingSet, validationSet);
        auto.setLogFeatureDir(DissertationConfig.getInstance().getProjectPath());
        auto.run();
    }
}
