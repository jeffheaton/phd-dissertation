package com.jeffheaton.dissertation.experiments.misc;

import com.jeffheaton.dissertation.JeffDissertation;
import com.jeffheaton.dissertation.autofeatures.AutoEngineerFeatures;
import com.jeffheaton.dissertation.autofeatures.Transform;
import com.jeffheaton.dissertation.experiments.manager.DissertationConfig;
import com.jeffheaton.dissertation.util.QuickEncodeDataset;
import org.encog.Encog;
import org.encog.mathutil.error.ErrorCalculation;
import org.encog.mathutil.error.ErrorCalculationMode;
import org.encog.mathutil.error.NormalizedError;
import org.encog.mathutil.randomize.generate.MersenneTwisterGenerateRandom;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.importance.FeatureImportance;
import org.encog.ml.importance.FeatureRank;
import org.encog.ml.importance.PerturbationFeatureImportanceCalc;
import org.encog.ml.prg.EncogProgram;
import org.encog.ml.train.MLTrain;
import org.encog.ml.train.strategy.end.EarlyStoppingStrategy;
import org.encog.ml.train.strategy.end.StoppingStrategy;
import org.encog.neural.networks.BasicNetwork;
import org.encog.persist.source.ObtainFallbackStream;
import org.encog.persist.source.ObtainInputStream;
import org.encog.util.Format;
import org.encog.util.csv.CSVFormat;
import org.encog.util.simple.EncogUtility;

import java.util.List;

public class ExperimentAutoFeature {

    private static String[] names;

    private static void status(MLTrain train, StoppingStrategy earlyStop) {
        StringBuilder line = new StringBuilder();

        line.append("Epoch #");
        line.append(train.getIteration());
        line.append(" Train Error:");
        line.append(Format.formatDouble(train.getError(), 6));
        line.append(", Stagnant: ");
        line.append(earlyStop.getStagnantIterations());
        System.out.println(line);
    }

    public static MLDataSet engineerFeatures() {
        ObtainInputStream source = new ObtainFallbackStream(
                DissertationConfig.getInstance().getDataPath().toString(),
                "auto-mpg.csv", JeffDissertation.class);

        QuickEncodeDataset quick = new QuickEncodeDataset(false,false);
        quick.analyze(source,"mpg", true, CSVFormat.EG_FORMAT);
        MLDataSet dataset = quick.generateDataset();

        AutoEngineerFeatures engineer = new AutoEngineerFeatures(dataset);

        engineer.setNames(quick.getFieldNames());
        //engineer.getDumpFeatures().setLogFeatureDir(DissertationConfig.getInstance().getProjectPath());
        engineer.setLogFeatureDir(DissertationConfig.getInstance().getProjectPath());
        engineer.process();

        System.out.println("Engineered features:");
        List<EncogProgram> engineeredFeatures = engineer.getFeatures(5);
        for(EncogProgram prg: engineeredFeatures) {
            System.out.println(prg.getScore() + ":" + prg.dumpAsCommonExpression());
        }

        MLDataSet augmentedDataset = engineer.augmentDataset(5,dataset);

        // Define the names of the columns of the augmented dataset.
        names = new String[augmentedDataset.getInputSize()];
        String[] origNames = quick.getFieldNames();
        int i=0;
        for(; i<origNames.length;i++) {
            names[i] = origNames[i];
        }
        int i2 = 1;
        while(i<names.length) {
            names[i++] = "FE #" + (i2++);
        }
        return augmentedDataset;
    }

    public static void trainAugmented(MLDataSet augmentedDataset) {
        Transform.zscore(augmentedDataset);

        // split
        MLDataSet[] split = EncogUtility.splitTrainValidate(augmentedDataset,new MersenneTwisterGenerateRandom(42),0.75);
        MLDataSet trainingSet = split[0];
        MLDataSet validationSet = split[1];

        // create a neural network, without using a factory
        BasicNetwork network = JeffDissertation.factorNeuralNetwork(trainingSet.getInputSize(),
                trainingSet.getIdealSize(), true);

        // train the neural network
        JeffDissertation.DissertationNeuralTraining d = JeffDissertation.factorNeuralTrainer(
                network,trainingSet,validationSet);
        MLTrain train = d.getTrain();
        EarlyStoppingStrategy earlyStop = d.getEarlyStop();

        int epoch = 1;

        do {
            train.iteration();
            System.out.println("Epoch #" + epoch + " Train Error:" + Format.formatDouble(train.getError(),6)
                    + ", Validation Error: " + Format.formatDouble(earlyStop.getValidationError(),6) +
                    ", Stagnant: " + earlyStop.getStagnantIterations());

            epoch++;
        } while(!train.isTrainingDone());
        train.finishTraining();

        network = (BasicNetwork) earlyStop.getBestModel();
        System.out.println("Best score (RMSE): " + earlyStop.getBestValidationError());

        NormalizedError error = new NormalizedError(validationSet);
        double normalizedError = error.calculateNormalizedRange(validationSet, network);
        System.out.println("Best score (normalized): " + normalizedError);

        System.out.println();
        System.out.println("Feature importance (permutation)");
        FeatureImportance fi = new PerturbationFeatureImportanceCalc(); //new NeuralFeatureImportanceCalc();
        fi.init(network,names);
        fi.performRanking(validationSet);

        for (FeatureRank ranking : fi.getFeaturesSorted()) {
            System.out.println(ranking.toString());
        }
        System.out.println(fi.toString());
    }

    public static void main(String[] args) {
        ErrorCalculation.setMode(ErrorCalculationMode.RMS);
        MLDataSet augmentedDataset = engineerFeatures();
        trainAugmented(augmentedDataset);

        Encog.getInstance().shutdown();
    }
}
