package com.jeffheaton.dissertation.experiments.misc;

import com.jeffheaton.dissertation.JeffDissertation;
import com.jeffheaton.dissertation.autofeatures.AutoEngineerFeatures;
import com.jeffheaton.dissertation.autofeatures.Transform;
import com.jeffheaton.dissertation.experiments.manager.DissertationConfig;
import com.jeffheaton.dissertation.experiments.manager.ExperimentTask;
import com.jeffheaton.dissertation.util.QuickEncodeDataset;
import org.encog.Encog;
import org.encog.engine.network.activation.ActivationLinear;
import org.encog.engine.network.activation.ActivationReLU;
import org.encog.mathutil.error.ErrorCalculation;
import org.encog.mathutil.error.ErrorCalculationMode;
import org.encog.mathutil.randomize.XaiverRandomizer;
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
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.sgd.StochasticGradientDescent;
import org.encog.neural.networks.training.propagation.sgd.update.AdamUpdate;
import org.encog.persist.source.ObtainFallbackStream;
import org.encog.persist.source.ObtainInputStream;
import org.encog.persist.source.ObtainResourceInputStream;
import org.encog.util.Format;
import org.encog.util.csv.CSVFormat;
import org.encog.util.simple.EncogUtility;

import java.io.File;
import java.util.List;

public class ExperimentAutoFeature {

    public static final int STAGNANT_STEPS = 500;
    public static final int MINI_BATCH_SIZE = 32;
    public static final double LEARNING_RATE = 1e-2;
    public static final double L1 = 0;
    public static final double L2 = 1e-8;

    public static int STAGNANT_AUTO = 50;

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


        // split
        MLDataSet[] split = EncogUtility.splitTrainValidate(dataset,new MersenneTwisterGenerateRandom(42),0.75);
        MLDataSet trainingSet = split[0];
        MLDataSet validationSet = split[1];


        AutoEngineerFeatures train = new AutoEngineerFeatures(trainingSet, validationSet);
        train.setNames(quick.getFieldNames());
        train.getDumpFeatures().setLogFeatureDir(DissertationConfig.getInstance().getProjectPath());
        train.setLogFeatureDir(DissertationConfig.getInstance().getProjectPath());
        StoppingStrategy stop = new StoppingStrategy(STAGNANT_AUTO);
        train.addStrategy(stop);

        do {
            train.iteration();

            status(train, stop);

            if (Double.isNaN(train.getError()) || Double.isInfinite(train.getError())) {
                break;
            }

        } while (!train.isTrainingDone());
        train.finishTraining();

        System.out.println("Engineered features:");
        List<EncogProgram> engineeredFeatures = train.getFeatures(5);
        for(EncogProgram prg: engineeredFeatures) {
            System.out.println(prg.getScore() + ":" + prg.dumpAsCommonExpression());
        }

        MLDataSet augmentedDataset = train.augmentDataset(5,dataset);

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
        BasicNetwork network = new BasicNetwork();
        network.addLayer(new BasicLayer(null,true,augmentedDataset.getInputSize()));
        network.addLayer(new BasicLayer(new ActivationReLU(),true,500));
        //network.addLayer(new BasicLayer(new ActivationReLU(),true,50));
        //network.addLayer(new BasicLayer(new ActivationReLU(),true,15));
        network.addLayer(new BasicLayer(new ActivationLinear(),false,trainingSet.getIdealSize()));
        network.getStructure().finalizeStructure();
        (new XaiverRandomizer()).randomize(network);
        //seedInput(network);

        // train the neural network
        StochasticGradientDescent train = new StochasticGradientDescent(network, trainingSet);
        train.setUpdateRule(new AdamUpdate());
        train.setBatchSize(MINI_BATCH_SIZE);
        train.setL1(L1);
        train.setL2(L2);
        train.setLearningRate(LEARNING_RATE);

        EarlyStoppingStrategy earlyStop = new EarlyStoppingStrategy(validationSet,5,STAGNANT_STEPS);
        earlyStop.setSaveBest(true);
        train.addStrategy(earlyStop);

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
        System.out.println("Best score: " + earlyStop.getBestValidationError());

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
