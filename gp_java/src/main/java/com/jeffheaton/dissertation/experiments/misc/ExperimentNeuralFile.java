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
import org.encog.neural.error.CrossEntropyErrorFunction;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.back.Backpropagation;
import org.encog.util.Format;
import org.encog.util.csv.CSVFormat;


public class ExperimentNeuralFile {

    public void runNeural() {
        ErrorCalculation.setMode(ErrorCalculationMode.RMS);

        ObtainInputStream source = new ObtainResourceInputStream("/auto-mpg.csv");
        QuickEncodeDataset quick = new QuickEncodeDataset();
        MLDataSet dataset = quick.process(source, "mpg", null, true, CSVFormat.EG_FORMAT);
        Transform.interpolate(dataset);
        Transform.zscore(dataset);

        // split
        MLDataSet[] split = Transform.splitTrainValidate(dataset,new MersenneTwisterGenerateRandom(42),0.75);
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
        final Backpropagation train = new Backpropagation(network, trainingSet, 1e-5, 0.9);
        //final ResilientPropagation train = new ResilientPropagation(network, trainingSet);
        train.setErrorFunction(new CrossEntropyErrorFunction());
        train.setNesterovUpdate(true);
        NewSimpleEarlyStoppingStrategy earlyStop = new NewSimpleEarlyStoppingStrategy(validationSet);
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


        NeuralFeatureImportanceCalc fi = new NeuralFeatureImportanceCalc(network,new String[] {
                "cylinders",
                "displacement",
                "horsepower",
                "weight",
                "acceleration",
                "year",
                "origin"
        });
        fi.calculateFeatureImportance();
        for (FeatureRanking ranking : fi.getFeatures()) {
            System.out.println(ranking.toString());
        }
        Encog.getInstance().shutdown();


    }

    public static void main(String[] args) {
        ExperimentNeuralFile prg = new ExperimentNeuralFile();
        prg.runNeural();
    }
}
