package com.jeffheaton.dissertation.experiments.manager;

import com.jeffheaton.dissertation.experiments.ExperimentResult;
import com.jeffheaton.dissertation.util.*;
import org.encog.EncogError;
import org.encog.engine.network.activation.ActivationLinear;
import org.encog.engine.network.activation.ActivationReLU;
import org.encog.mathutil.randomize.XaiverRandomizer;
import org.encog.mathutil.randomize.generate.GenerateRandom;
import org.encog.mathutil.randomize.generate.MersenneTwisterGenerateRandom;
import org.encog.ml.data.MLDataSet;
import org.encog.neural.error.CrossEntropyErrorFunction;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.sgd.StochasticGradientDescent;
import org.encog.util.Format;
import org.encog.util.Stopwatch;
import org.encog.util.csv.CSVFormat;

/**
 * Created by jeff on 5/16/16.
 */
public class ExperimentTask implements Runnable {

    private final String name;
    private final String algorithm;
    private final String dataset;
    private final int cycle;
    private String status = "queued";

    public ExperimentTask(String theName, String theDataset, String theAlgorithm, int theCycle) {
        this.name = theName;
        this.dataset = theDataset;
        this.algorithm = theAlgorithm;
        this.cycle = theCycle;
    }

    public String getName() {
        return name;
    }

    public String getAlgorithm() {
        return algorithm;
    }

    public String getDataset() {
        return dataset;
    }

    public int getCycle() {
        return cycle;
    }

    public String getKey() {
        StringBuilder result = new StringBuilder();
        result.append(this.name);
        result.append("|");
        result.append(this.algorithm);
        result.append("|");
        result.append(this.dataset);
        result.append("|");
        result.append(this.cycle);
        return result.toString();
    }

    private void verboseStatus(int cycle, StochasticGradientDescent train, NewSimpleEarlyStoppingStrategy earlyStop) {
        System.out.println("Cycle #"+(cycle+1)+",Epoch #" + train.getIteration() + " Train Error:"
                + Format.formatDouble(train.getError(), 6)
                + ", Validation Error: " + Format.formatDouble(earlyStop.getValidationError(), 6) +
                ", Stagnant: " + earlyStop.getStagnantIterations());
    }

    public void runNeural(MLDataSet dataset) {
            Stopwatch sw = new Stopwatch();
            sw.start();
            // split
            GenerateRandom rnd = new MersenneTwisterGenerateRandom(42);
            org.encog.ml.data.MLDataSet[] split = Transform.splitTrainValidate(dataset,rnd,0.75);
            MLDataSet trainingSet = split[0];
            MLDataSet validationSet = split[1];

            // create a neural network, without using a factory
            BasicNetwork network = new BasicNetwork();
            network.addLayer(new BasicLayer(null,true,trainingSet.getInputSize()));
            network.addLayer(new BasicLayer(new ActivationReLU(),true,400));
            network.addLayer(new BasicLayer(new ActivationReLU(),true,200));
            network.addLayer(new BasicLayer(new ActivationReLU(),true,100));
            network.addLayer(new BasicLayer(new ActivationReLU(),true,50));
            network.addLayer(new BasicLayer(new ActivationReLU(),true,25));
            network.addLayer(new BasicLayer(new ActivationLinear(),false,trainingSet.getIdealSize()));
            network.getStructure().finalizeStructure();
            (new XaiverRandomizer()).randomize(network);


            // train the neural network
            final StochasticGradientDescent train = new StochasticGradientDescent(network, trainingSet, 100, 1e-6, 0.9);
            train.setErrorFunction(new CrossEntropyErrorFunction());

            NewSimpleEarlyStoppingStrategy earlyStop = new NewSimpleEarlyStoppingStrategy(validationSet);
            train.addStrategy(earlyStop);

            long lastUpdate = System.currentTimeMillis();

            do {
                train.iteration();

                long sinceLastUpdate = (System.currentTimeMillis() - lastUpdate)/1000;

                if( train.getIteration()==1 || train.isTrainingDone() || sinceLastUpdate>60 ) {
                    verboseStatus(cycle, train, earlyStop);
                    lastUpdate = System.currentTimeMillis();
                }
            } while(!train.isTrainingDone());
            train.finishTraining();

            sw.stop();
            System.out.println(earlyStop.getValidationError());

            //result.addResult(earlyStop.getValidationError(),sw.getElapsedMilliseconds());
        }


    public void run() {
        MLDataSet dataset = null;

        if( this.dataset.equals("mpg")) {
            ObtainInputStream source = new ObtainResourceInputStream("/auto-mpg.csv");
            QuickEncodeDataset quick = new QuickEncodeDataset();
            dataset = quick.process(source,0, true, CSVFormat.EG_FORMAT);
            Transform.interpolate(dataset);
        } else {
            throw new EncogError("Unknown dataset: " + this.dataset);
        }


        if( this.algorithm.equalsIgnoreCase("neural")) {
            runNeural(dataset);
        } else {
            throw new EncogError("Unknown algorithm: " + this.dataset);
        }

    }

    public String getStatus() {
        return status;
    }

    public void setStatus(String status) {
        this.status = status;
    }

    @Override
    public String toString() {
        StringBuilder result = new StringBuilder();
        result.append("[ExperimentTask:");
        result.append(getKey());
        result.append("]");
        return result.toString();
    }

    public boolean isQueued() {
        return this.status.equalsIgnoreCase("queued");
    }

    public void claim(String owner) {
        this.status = "running-" + owner;
    }

    public void reportDone(String owner) {
        this.status = "done-" + owner;
    }

    public boolean isComplete() {
        return this.status.startsWith("done");
    }
}
