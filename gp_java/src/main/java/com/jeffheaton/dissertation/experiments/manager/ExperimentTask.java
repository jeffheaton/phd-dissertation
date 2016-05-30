package com.jeffheaton.dissertation.experiments.manager;

import com.jeffheaton.dissertation.experiments.ExperimentResult;
import com.jeffheaton.dissertation.util.*;
import org.encog.EncogError;
import org.encog.engine.network.activation.ActivationLinear;
import org.encog.engine.network.activation.ActivationReLU;
import org.encog.mathutil.error.ErrorCalculation;
import org.encog.mathutil.error.ErrorCalculationMode;
import org.encog.mathutil.randomize.XaiverRandomizer;
import org.encog.mathutil.randomize.generate.GenerateRandom;
import org.encog.mathutil.randomize.generate.MersenneTwisterGenerateRandom;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.ea.score.adjust.ComplexityAdjustedScore;
import org.encog.ml.ea.train.basic.TrainEA;
import org.encog.ml.fitness.MultiObjectiveFitness;
import org.encog.ml.prg.EncogProgram;
import org.encog.ml.prg.EncogProgramContext;
import org.encog.ml.prg.PrgCODEC;
import org.encog.ml.prg.extension.FunctionFactory;
import org.encog.ml.prg.extension.StandardExtensions;
import org.encog.ml.prg.generator.RampedHalfAndHalf;
import org.encog.ml.prg.opp.ConstMutation;
import org.encog.ml.prg.opp.SubtreeCrossover;
import org.encog.ml.prg.opp.SubtreeMutation;
import org.encog.ml.prg.species.PrgSpeciation;
import org.encog.ml.prg.train.PrgPopulation;
import org.encog.ml.prg.train.rewrite.RewriteAlgebraic;
import org.encog.ml.prg.train.rewrite.RewriteConstants;
import org.encog.neural.error.CrossEntropyErrorFunction;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.TrainingSetScore;
import org.encog.neural.networks.training.propagation.sgd.StochasticGradientDescent;
import org.encog.parse.expression.latex.RenderLatexExpression;
import org.encog.util.Format;
import org.encog.util.Stopwatch;
import org.encog.util.csv.CSVFormat;

import java.util.Random;

/**
 * Created by jeff on 5/16/16.
 */
public class ExperimentTask implements Runnable {

    private final String name;
    private final String algorithm;
    private final String dataset;
    private final int cycle;
    private String status = "queued";
    private int iterations;
    private double result;
    private int elapsed;

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

    public void runGP(QuickEncodeDataset quick, MLDataSet dataset) {
        Stopwatch sw = new Stopwatch();
        ErrorCalculation.setMode(ErrorCalculationMode.RMS);
        sw.start();
        // split
        GenerateRandom rnd = new MersenneTwisterGenerateRandom(42);
        org.encog.ml.data.MLDataSet[] split = Transform.splitTrainValidate(dataset,rnd,0.75);
        MLDataSet trainingSet = split[0];
        MLDataSet validationSet = split[1];

        EncogProgramContext context = new EncogProgramContext();
        for(int i=0;i<quick.getName().length;i++) {
            if( i!=quick.getTargetColumn() && quick.getNumeric()[i] ) {
                context.defineVariable(quick.getName()[i]);
            }
        }

        //StandardExtensions.createNumericOperators(context);

        FunctionFactory factory = context.getFunctions();
        factory.addExtension(StandardExtensions.EXTENSION_VAR_SUPPORT);
        factory.addExtension(StandardExtensions.EXTENSION_CONST_SUPPORT);
        factory.addExtension(StandardExtensions.EXTENSION_NEG);
        factory.addExtension(StandardExtensions.EXTENSION_ADD);
        factory.addExtension(StandardExtensions.EXTENSION_SUB);
        factory.addExtension(StandardExtensions.EXTENSION_MUL);
        factory.addExtension(StandardExtensions.EXTENSION_DIV);
        factory.addExtension(StandardExtensions.EXTENSION_POWER);



        PrgPopulation pop = new PrgPopulation(context,1000);

        MultiObjectiveFitness score = new MultiObjectiveFitness();
        score.addObjective(1.0, new TrainingSetScore(trainingSet));

        TrainEA genetic = new TrainEA(pop, score);
        //genetic.setValidationMode(true);
        genetic.setCODEC(new PrgCODEC());
        genetic.addOperation(0.5, new SubtreeCrossover());
        genetic.addOperation(0.25, new ConstMutation(context,0.5,1.0));
        genetic.addOperation(0.25, new SubtreeMutation(context,4));
        genetic.addScoreAdjuster(new ComplexityAdjustedScore(10,20,10,50.0));
        genetic.getRules().addRewriteRule(new RewriteConstants());
        genetic.getRules().addRewriteRule(new RewriteAlgebraic());
        genetic.setSpeciation(new PrgSpeciation());
        genetic.setThreadCount(1);

        NewSimpleEarlyStoppingStrategy earlyStop = new NewSimpleEarlyStoppingStrategy(validationSet, 5, 500, 0.01);
        genetic.addStrategy(earlyStop);

        (new RampedHalfAndHalf(context,1, 6)).generate(new Random(), pop);

        genetic.setShouldIgnoreExceptions(false);

        EncogProgram best = null;


            do {
                genetic.iteration();
                best = (EncogProgram) genetic.getBestGenome();
                System.out.println(genetic.getIteration() + ", Error: "
                        + Format.formatDouble(best.getScore(),6) + ",Validation Score: " + earlyStop.getValidationError()
                        + ",Best Genome Size:" +best.size()
                        + ",Species Count:" + pop.getSpecies().size() + ",best: " + best.dumpAsCommonExpression());
            } while(!genetic.isTrainingDone());

            //EncogUtility.evaluate(best, trainingData);

            System.out.println("Final score:" + best.getScore()
                    + ", effective score:" + best.getAdjustedScore());
            System.out.println(best.dumpAsCommonExpression());
            System.out.println();
            //pop.dumpMembers(Integer.MAX_VALUE);
            //pop.dumpMembers(10);

        this.elapsed = (int)(sw.getElapsedMilliseconds()/1000);
        this.result = earlyStop.getValidationError();
        this.iterations = genetic.getIteration();
    }

    public void runNeural(MLDataSet dataset) {
            Stopwatch sw = new Stopwatch();
            ErrorCalculation.setMode(ErrorCalculationMode.RMS);
            sw.start();
            // split
            GenerateRandom rnd = new MersenneTwisterGenerateRandom(42);
            org.encog.ml.data.MLDataSet[] split = Transform.splitTrainValidate(dataset,rnd,0.75);
            MLDataSet trainingSet = split[0];
            MLDataSet validationSet = split[1];

            // create a neural network, without using a factory
            BasicNetwork network = new BasicNetwork();
            network.addLayer(new BasicLayer(null,true,trainingSet.getInputSize()));
            network.addLayer(new BasicLayer(new ActivationReLU(),true,100));
            //network.addLayer(new BasicLayer(new ActivationReLU(),true,200));
            //network.addLayer(new BasicLayer(new ActivationReLU(),true,100));
            //network.addLayer(new BasicLayer(new ActivationReLU(),true,50));
            network.addLayer(new BasicLayer(new ActivationReLU(),true,25));
            network.addLayer(new BasicLayer(new ActivationLinear(),false,trainingSet.getIdealSize()));
            network.getStructure().finalizeStructure();
            (new XaiverRandomizer()).randomize(network);


            // train the neural network
            final StochasticGradientDescent train = new StochasticGradientDescent(network, trainingSet, 100, 1e-6, 0.9);
            train.setErrorFunction(new CrossEntropyErrorFunction());
            train.setThreadCount(1);

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
            this.elapsed = (int)(sw.getElapsedMilliseconds()/1000);
            this.result = earlyStop.getValidationError();
            this.iterations = train.getIteration();
        }


    public void run() {
        MLDataSet dataset = null;
        QuickEncodeDataset quick;

        if( this.dataset.equals("autompg")) {
            ObtainInputStream source = new ObtainResourceInputStream("/auto-mpg.csv");
            quick = new QuickEncodeDataset();
            dataset = quick.process(source,0, true, CSVFormat.EG_FORMAT);
            Transform.interpolate(dataset);
            Transform.zscore(dataset);
        } else if( this.dataset.equals("iris")) {
            ObtainInputStream source = new ObtainResourceInputStream("/iris.csv");
            quick = new QuickEncodeDataset();
            dataset = quick.process(source,0, true, CSVFormat.EG_FORMAT);
            Transform.interpolate(dataset);
            Transform.zscore(dataset);
        } else {
            throw new EncogError("Unknown dataset: " + this.dataset);
        }

        if( this.algorithm.equalsIgnoreCase("neural")) {
            runNeural(dataset);
        } else if( this.algorithm.equalsIgnoreCase("gp")) {
            runGP(quick, dataset);
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

    public void reportDone(String theOwner) {
        this.status = "done-" + theOwner;
    }

    public boolean isComplete() {
        return this.status.startsWith("done") || this.status.startsWith("error");
    }

    public void reportError(String owner, Exception ex) {
        this.status = "error-" + owner;
    }

    public int getIterations() {
        return iterations;
    }

    public void setIterations(int iterations) {
        this.iterations = iterations;
    }

    public double getResult() {
        return result;
    }

    public void setResult(double result) {
        this.result = result;
    }

    public int getElapsed() {
        return elapsed;
    }

    public void setElapsed(int elapsed) {
        this.elapsed = elapsed;
    }
}
