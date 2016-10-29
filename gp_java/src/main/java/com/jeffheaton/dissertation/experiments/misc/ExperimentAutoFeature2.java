package com.jeffheaton.dissertation.experiments.misc;

import com.jeffheaton.dissertation.JeffDissertation;
import com.jeffheaton.dissertation.autofeatures.AutoEngineerFeatures;
import com.jeffheaton.dissertation.autofeatures.AutoFeatureGP;
import com.jeffheaton.dissertation.experiments.manager.DissertationConfig;
import com.jeffheaton.dissertation.experiments.manager.ExperimentTask;
import com.jeffheaton.dissertation.util.QuickEncodeDataset;
import org.encog.mathutil.error.ErrorCalculation;
import org.encog.mathutil.error.ErrorCalculationMode;
import org.encog.mathutil.randomize.generate.MersenneTwisterGenerateRandom;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.train.MLTrain;
import org.encog.ml.train.strategy.end.EarlyStoppingStrategy;
import org.encog.ml.train.strategy.end.EndIterationsStrategy;
import org.encog.ml.train.strategy.end.StoppingStrategy;
import org.encog.persist.source.ObtainFallbackStream;
import org.encog.persist.source.ObtainInputStream;
import org.encog.persist.source.ObtainResourceInputStream;
import org.encog.util.Format;
import org.encog.util.csv.CSVFormat;
import org.encog.util.simple.EncogUtility;

import java.io.File;

public class ExperimentAutoFeature2 {

    public static int STAGNANT_AUTO = 50;

    private static void status(MLTrain train, StoppingStrategy earlyStop) {
        ErrorCalculation.setMode(ErrorCalculationMode.RMS);
        StringBuilder line = new StringBuilder();

        line.append("Epoch #");
        line.append(train.getIteration());
        line.append(" Train Error:");
        line.append(Format.formatDouble(train.getError(), 6));
        line.append(", Stagnant: ");
        line.append(earlyStop.getStagnantIterations());
        System.out.println(line);
    }

    public static void main(String[] args) {
        ErrorCalculation.setMode(ErrorCalculationMode.RMS);

        ObtainInputStream source = new ObtainFallbackStream(
                DissertationConfig.getInstance().getDataPath().toString(),
                "auto-mpg.csv", JeffDissertation.class);

        QuickEncodeDataset quick = new QuickEncodeDataset(false,false);
        quick.analyze(source,"mpg", true, CSVFormat.EG_FORMAT);
        MLDataSet dataset = quick.generateDataset();

        AutoFeatureGP train = new AutoFeatureGP(dataset);
        train.getDumpFeatures().setLogFeatureDir(DissertationConfig.getInstance().getProjectPath());
        StoppingStrategy stop = new StoppingStrategy(STAGNANT_AUTO);
        train.addStrategy(stop);
        train.addStrategy(new EndIterationsStrategy(5));

        do {
            train.iteration();

            status(train, stop);

            if (Double.isNaN(train.getError()) || Double.isInfinite(train.getError())) {
                break;
            }

        } while (!train.isTrainingDone());
        train.finishTraining();
    }
}
