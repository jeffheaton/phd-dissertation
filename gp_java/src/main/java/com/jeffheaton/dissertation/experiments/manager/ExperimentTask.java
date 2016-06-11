package com.jeffheaton.dissertation.experiments.manager;

import com.jeffheaton.dissertation.util.*;
import org.encog.EncogError;
import org.encog.engine.network.activation.ActivationLinear;
import org.encog.engine.network.activation.ActivationReLU;
import org.encog.engine.network.activation.ActivationSoftMax;
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
import org.encog.ml.train.MLTrain;
import org.encog.neural.error.CrossEntropyErrorFunction;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.TrainingSetScore;
import org.encog.neural.networks.training.propagation.back.Backpropagation;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;
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

    public static final int MINI_BATCH_SIZE = 50;
    public static final double LEARNING_RATE = 1e-12;
    public static final double MOMENTUM = 0.9;
    public static final int POPULATION_SIZE = 100;
    public static final int STAGNANT_NEURAL = 50;

    private final String name;
    private final String algorithm;
    private final String datasetFilename;
    private final int cycle;
    private String status = "queued";
    private int iterations;
    private double result;
    private int elapsed;
    private QuickEncodeDataset quick;
    private MLDataSet dataset;
    private ThreadedRunner owner;
    private String predictors;
    private String info;

    public ExperimentTask(String theName, String theDataset, String theAlgorithm, String thePredictors, int theCycle) {
        this.name = theName;
        this.datasetFilename = theDataset;
        this.algorithm = theAlgorithm;
        this.cycle = theCycle;
        this.predictors = thePredictors;
    }

    public String getName() {
        return name;
    }

    public String getAlgorithm() {
        return algorithm;
    }

    public String getDatasetFilename() {
        return this.datasetFilename;
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
        result.append(this.datasetFilename);
        result.append("|");
        result.append(this.cycle);
        return result.toString();
    }

    private void verboseStatusNeural(int cycle, MLTrain train, NewSimpleEarlyStoppingStrategy earlyStop) {
        if( this.owner==null || this.owner.isVerbose() ) {
            System.out.println("Cycle #" + (cycle + 1) + ",Epoch #" + train.getIteration() + " Train Error:"
                    + Format.formatDouble(train.getError(), 6)
                    + ", Validation Error: " + Format.formatDouble(earlyStop.getValidationError(), 6) +
                    ", Stagnant: " + earlyStop.getStagnantIterations());
        }
    }

    public void verboseStatusGeneticProgram(TrainEA genetic, EncogProgram best, NewSimpleEarlyStoppingStrategy earlyStop, PrgPopulation pop) {
        if( this.owner==null || this.owner.isVerbose() ) {
            System.out.println(genetic.getIteration() + ", Error: "
                    + Format.formatDouble(best.getScore(), 6) + ",Validation Score: " + earlyStop.getValidationError()
                    + ",Best Genome Size:" + best.size()
                    + ",Species Count:" + pop.getSpecies().size() + ",best: " + best.dumpAsCommonExpression());
        }
    }

    public void runGP(MLDataSet dataset, boolean regression) {

        if(!regression) {
            throw new EncogError("Cannot currently evaluate GP classification.");
        }

        Stopwatch sw = new Stopwatch();
        ErrorCalculation.setMode(ErrorCalculationMode.RMS);
        sw.start();
        // split
        GenerateRandom rnd = new MersenneTwisterGenerateRandom(42);
        org.encog.ml.data.MLDataSet[] split = Transform.splitTrainValidate(dataset, rnd, 0.75);
        MLDataSet trainingSet = split[0];
        MLDataSet validationSet = split[1];

        EncogProgramContext context = new EncogProgramContext();
        for (QuickEncodeDataset.QuickField field: quick.getPredictors()) {
            context.defineVariable(field.getName());
        }

        FunctionFactory factory = context.getFunctions();
        factory.addExtension(StandardExtensions.EXTENSION_VAR_SUPPORT);
        factory.addExtension(StandardExtensions.EXTENSION_CONST_SUPPORT);
        factory.addExtension(StandardExtensions.EXTENSION_NEG);
        factory.addExtension(StandardExtensions.EXTENSION_ADD);
        factory.addExtension(StandardExtensions.EXTENSION_SUB);
        factory.addExtension(StandardExtensions.EXTENSION_MUL);
        factory.addExtension(StandardExtensions.EXTENSION_DIV);
        factory.addExtension(StandardExtensions.EXTENSION_POWER);


        PrgPopulation pop = new PrgPopulation(context, POPULATION_SIZE);

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

        NewSimpleEarlyStoppingStrategy earlyStop = new NewSimpleEarlyStoppingStrategy(validationSet, 5, 50, 0.01);
        genetic.addStrategy(earlyStop);

        (new RampedHalfAndHalf(context, 1, 6)).generate(new Random(), pop);

        genetic.setShouldIgnoreExceptions(false);

        EncogProgram best = null;
        long lastUpdate = System.currentTimeMillis();


        do {
            long sinceLastUpdate = (System.currentTimeMillis() - lastUpdate) / 1000;
            genetic.iteration();
            best = (EncogProgram) genetic.getBestGenome();
            if (this.owner==null || genetic.getIteration() == 1 || genetic.isTrainingDone() || sinceLastUpdate > 60) {
                verboseStatusGeneticProgram(genetic, best, earlyStop, pop);
            }
        } while (!genetic.isTrainingDone());

        this.elapsed = (int) (sw.getElapsedMilliseconds() / 1000);
        this.result = earlyStop.getValidationError();
        this.iterations = genetic.getIteration();

        setInfo(best.dumpAsCommonExpression());
    }

    public void runNeural(MLDataSet dataset, boolean regression) {
        Stopwatch sw = new Stopwatch();
        sw.start();
        // split
        GenerateRandom rnd = new MersenneTwisterGenerateRandom(42);
        org.encog.ml.data.MLDataSet[] split = Transform.splitTrainValidate(dataset, rnd, 0.75);
        MLDataSet trainingSet = split[0];
        MLDataSet validationSet = split[1];

        // create a neural network, without using a factory
        BasicNetwork network = new BasicNetwork();
        network.addLayer(new BasicLayer(null, true, trainingSet.getInputSize()));
        network.addLayer(new BasicLayer(new ActivationReLU(), true, 200));
        network.addLayer(new BasicLayer(new ActivationReLU(),true,100));
        network.addLayer(new BasicLayer(new ActivationReLU(), true, 25));

        if (regression) {
            network.addLayer(new BasicLayer(new ActivationLinear(), false, trainingSet.getIdealSize()));
            ErrorCalculation.setMode(ErrorCalculationMode.RMS);
        } else {
            network.addLayer(new BasicLayer(new ActivationSoftMax(), false, trainingSet.getIdealSize()));
            ErrorCalculation.setMode(ErrorCalculationMode.HOT_LOGLOSS);
        }
        network.getStructure().finalizeStructure();
        (new XaiverRandomizer()).randomize(network);
        //network.reset();

        // train the neural network
        int miniBatchSize = Math.min(dataset.size(),MINI_BATCH_SIZE);
        double learningRate = LEARNING_RATE / miniBatchSize;
        MiniBatchDataSet batchedDataSet = new MiniBatchDataSet(trainingSet,rnd);
        batchedDataSet.setBatchSize(miniBatchSize);
        Backpropagation train = new Backpropagation(network, trainingSet, learningRate, MOMENTUM);
        train.setErrorFunction(new CrossEntropyErrorFunction());
        train.setThreadCount(1);

        NewSimpleEarlyStoppingStrategy earlyStop = new NewSimpleEarlyStoppingStrategy(validationSet,10,STAGNANT_NEURAL,0.01);
        train.addStrategy(earlyStop);

        long lastUpdate = System.currentTimeMillis();

        do {
            train.iteration();
            batchedDataSet.advance();

            long sinceLastUpdate = (System.currentTimeMillis() - lastUpdate) / 1000;

            if (this.owner==null || train.getIteration() == 1 || train.isTrainingDone() || sinceLastUpdate > 60) {
                verboseStatusNeural(cycle, train, earlyStop);
                lastUpdate = System.currentTimeMillis();
            }

            if( Double.isNaN(train.getError()) || Double.isInfinite(train.getError()) ) {
                break;
            }

        } while (!train.isTrainingDone());
        train.finishTraining();

        sw.stop();
        this.elapsed = (int) (sw.getElapsedMilliseconds() / 1000);
        this.result = earlyStop.getValidationError();
        this.iterations = train.getIteration();

        setInfo("Neural network done.");
    }

    private void loadDataset(boolean singleFieldCatagorical, String target) {
        ObtainInputStream source = new ObtainFallbackStream(this.datasetFilename);
        this.quick = new QuickEncodeDataset(singleFieldCatagorical,false);
        this.quick.dumpFieldInfo();
        this.dataset = quick.process(source, target, this.predictors, true, CSVFormat.EG_FORMAT);
        //Transform.interpolate(dataset);
    }

    public void run() {
        ParseModelType model = new ParseModelType(this.algorithm);

        if (model.isNeuralNetwork()) {
            loadDataset(false,model.getTarget());
            runNeural(dataset, model.isRegression());
        } else if (model.isGeneticProgram()) {
            loadDataset(true,model.getTarget());
            runGP( dataset, model.isRegression());
        } else {
            throw new EncogError("Unknown algorithm: " + this.algorithm);
        }
        System.out.println("Complete: " + getKey() + " - " + this.result + " - ");

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

    public ThreadedRunner getOwner() {
        return owner;
    }

    public void setOwner(ThreadedRunner owner) {
        this.owner = owner;
    }

    public String getPredictors() {
        return this.predictors;
    }

    public String getInfo() {
        return info;
    }

    public void setInfo(String info) {
        this.info = info;
    }
}
