package com.jeffheaton.dissertation.experiments.misc;

import com.jeffheaton.dissertation.util.*;
import org.encog.Encog;
import org.encog.engine.network.activation.ActivationLinear;
import org.encog.engine.network.activation.ActivationReLU;
import org.encog.mathutil.error.ErrorCalculation;
import org.encog.mathutil.error.ErrorCalculationMode;
import org.encog.mathutil.randomize.XaiverRandomizer;
import org.encog.mathutil.randomize.generate.MersenneTwisterGenerateRandom;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.importance.*;
import org.encog.ml.train.strategy.end.EarlyStoppingStrategy;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.sgd.StochasticGradientDescent;
import org.encog.util.Format;
import org.encog.util.csv.CSVFormat;
import org.encog.util.simple.EncogUtility;


public class ExperimentNeuralFile {

    public void runNeural() {
        ErrorCalculation.setMode(ErrorCalculationMode.RMS);

        ObtainInputStream source = new ObtainFallbackStream("auto-mpg.csv");
        QuickEncodeDataset quick = new QuickEncodeDataset(false,true);
        quick.analyze(source, "mpg", true, CSVFormat.EG_FORMAT);
        MLDataSet dataset = quick.generateDataset();

        // split
        MLDataSet[] split = EncogUtility.splitTrainValidate(dataset,new MersenneTwisterGenerateRandom(42),0.75);
        MLDataSet trainingSet = split[0];
        MLDataSet validationSet = split[1];

        // create a neural network, without using a factory
        BasicNetwork network = new BasicNetwork();
        network.addLayer(new BasicLayer(null,true,trainingSet.getInputSize()));
        network.addLayer(new BasicLayer(new ActivationReLU(),true,500));
        //network.addLayer(new BasicLayer(new ActivationReLU(),true,50));
        //network.addLayer(new BasicLayer(new ActivationReLU(),true,15));
        network.addLayer(new BasicLayer(new ActivationLinear(),false,trainingSet.getIdealSize()));
        network.getStructure().finalizeStructure();
        (new XaiverRandomizer()).randomize(network);
        //seedInput(network);

        // train the neural network
        final StochasticGradientDescent train = new StochasticGradientDescent(network, trainingSet);
        //train.setErrorFunction(new CrossEntropyErrorFunction());
        //train.setNesterovUpdate(true);
        train.setLearningRate(0.1);
        train.setL2(1e-7);

        EarlyStoppingStrategy earlyStop = new EarlyStoppingStrategy(validationSet);
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

        System.out.println();
        System.out.println("Feature importance (permutation)");
        FeatureImportance fi = new PermutationFeatureImportanceCalc(); //new NeuralFeatureImportanceCalc();
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
