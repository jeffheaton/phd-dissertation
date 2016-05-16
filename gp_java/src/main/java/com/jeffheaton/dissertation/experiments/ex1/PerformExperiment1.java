package com.jeffheaton.dissertation.experiments.ex1;

import com.jeffheaton.dissertation.experiments.ExperimentResult;
import com.jeffheaton.dissertation.experiments.data.SyntheticDatasets;
import com.jeffheaton.dissertation.util.NewSimpleEarlyStoppingStrategy;
import com.jeffheaton.dissertation.util.Transform;
import org.encog.Encog;
import org.encog.engine.network.activation.ActivationLinear;
import org.encog.engine.network.activation.ActivationReLU;
import org.encog.mathutil.error.ErrorCalculation;
import org.encog.mathutil.error.ErrorCalculationMode;
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

/**
 * Created by jeff on 5/10/16.
 */
public class PerformExperiment1 {

    public static void verboseStatus(int cycle, StochasticGradientDescent train, NewSimpleEarlyStoppingStrategy earlyStop) {
        System.out.println("Cycle #"+(cycle+1)+",Epoch #" + train.getIteration() + " Train Error:"
                + Format.formatDouble(train.getError(), 6)
                + ", Validation Error: " + Format.formatDouble(earlyStop.getValidationError(), 6) +
                ", Stagnant: " + earlyStop.getStagnantIterations());
    }

    public static void runCycle(int cycle, MLDataSet dataset, ExperimentResult result) {
        Stopwatch sw = new Stopwatch();
        sw.start();
        // split
        GenerateRandom rnd = new MersenneTwisterGenerateRandom(42);
        MLDataSet[] split = Transform.splitTrainValidate(dataset,rnd,0.75);
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

        result.addResult(earlyStop.getValidationError(),sw.getElapsedMilliseconds());
    }

    public static void runExperiment(MLDataSet dataset) {
        ExperimentResult result = new ExperimentResult("neural");
        for(int i=0;i<5;i++) {
            runCycle(i ,dataset, result);
        }
        System.out.println(result.toString());

    }

    public static void main(String[] args) {
        Stopwatch sw = new Stopwatch();
        sw.start();

        ErrorCalculation.setMode(ErrorCalculationMode.MSE);
        //MLDataSet dataset = SyntheticDatasets.generateDiffRatio();
        MLDataSet dataset = SyntheticDatasets.generatePolynomial();

        runExperiment(dataset);


        sw.stop();
        System.out.println("Total runtime: " + Format.formatTimeSpan((int)(sw.getElapsedMilliseconds()/1000)));

        Encog.getInstance().shutdown();


    }
}
