package com.jeffheaton.dissertation.experiments.misc;

import com.jeffheaton.dissertation.JeffDissertation;
import com.jeffheaton.dissertation.experiments.manager.DissertationConfig;
import com.jeffheaton.dissertation.util.*;
import org.encog.Encog;
import org.encog.mathutil.error.ErrorCalculation;
import org.encog.mathutil.error.ErrorCalculationMode;
import org.encog.mathutil.randomize.generate.MersenneTwisterGenerateRandom;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.importance.*;
import org.encog.ml.train.MLTrain;
import org.encog.ml.train.strategy.end.EarlyStoppingStrategy;
import org.encog.neural.networks.BasicNetwork;
import org.encog.persist.source.ObtainFallbackStream;
import org.encog.persist.source.ObtainInputStream;
import org.encog.util.Format;
import org.encog.util.csv.CSVFormat;
import org.encog.util.simple.EncogUtility;
import org.encog.ml.importance.PerturbationFeatureImportanceCalc;


public class ExperimentNeuralFile {

    public void runNeural() {
        ErrorCalculation.setMode(ErrorCalculationMode.RMS);

        ObtainInputStream source = new ObtainFallbackStream(
                DissertationConfig.getInstance().getDataPath().toString(),
                "auto-mpg.csv", JeffDissertation.class);
        QuickEncodeDataset quick = new QuickEncodeDataset(false,true);
        quick.analyze(source, "mpg", true, CSVFormat.EG_FORMAT);
        MLDataSet dataset = quick.generateDataset();

        // split
        MLDataSet[] split = EncogUtility.splitTrainValidate(dataset,new MersenneTwisterGenerateRandom(42),0.75);
        MLDataSet trainingSet = split[0];
        MLDataSet validationSet = split[1];

        // create a neural network, without using a factory
        BasicNetwork network = JeffDissertation.factorNeuralNetwork(
                trainingSet.getInputSize(),trainingSet.getIdealSize(),true);

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
        System.out.println("Best score: " + earlyStop.getBestValidationError());

        System.out.println();
        System.out.println("Feature importance (permutation)");
        FeatureImportance fi = new PerturbationFeatureImportanceCalc(); //new NeuralFeatureImportanceCalc();
        fi.init(network,quick.nameOutputVectorFields());
        fi.performRanking(validationSet);

        for (FeatureRank ranking : fi.getFeaturesSorted()) {
            System.out.println(ranking.toString());
        }
        System.out.println(fi.toString());

        System.out.println();
        System.out.println("Feature importance (weights)");
        fi = new NeuralFeatureImportanceCalc();
        fi.init(network,quick.nameOutputVectorFields());
        fi.performRanking(validationSet);

        for (FeatureRank ranking : fi.getFeaturesSorted()) {
            System.out.println(ranking.toString());
        }
        System.out.println(fi.toString());

        System.out.println();
        System.out.println("Feature importance (covariance, without neural network)");
        fi = new CorrelationFeatureImportanceCalc();
        fi.init(null,quick.nameOutputVectorFields());
        fi.performRanking(trainingSet);

        for (FeatureRank ranking : fi.getFeaturesSorted()) {
            System.out.println(ranking.toString());
        }
        System.out.println(fi.toString());

        Encog.getInstance().shutdown();


    }

    public static void main(String[] args) {
        ExperimentNeuralFile prg = new ExperimentNeuralFile();
        prg.runNeural();
    }
}
