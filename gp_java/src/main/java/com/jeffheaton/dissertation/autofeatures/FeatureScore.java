package com.jeffheaton.dissertation.autofeatures;

import org.encog.Encog;
import org.encog.EncogError;
import org.encog.engine.network.activation.ActivationLinear;
import org.encog.engine.network.activation.ActivationReLU;
import org.encog.mathutil.randomize.XaiverRandomizer;
import org.encog.ml.CalculateScore;
import org.encog.ml.MLMethod;
import org.encog.ml.MLRegression;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataPair;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.ml.ea.exception.EARuntimeError;
import org.encog.ml.ea.genome.Genome;
import org.encog.ml.ea.population.Population;
import org.encog.ml.importance.FeatureRank;
import org.encog.ml.importance.PerturbationFeatureImportanceCalc;
import org.encog.ml.train.MLTrain;
import org.encog.ml.train.strategy.end.EarlyStoppingStrategy;
import org.encog.ml.train.strategy.end.EndIterationsStrategy;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.back.Backpropagation;
import org.encog.neural.networks.training.propagation.sgd.StochasticGradientDescent;
import org.encog.neural.networks.training.propagation.sgd.update.AdaGradUpdate;
import org.encog.neural.networks.training.propagation.sgd.update.AdamUpdate;
import org.encog.util.Format;

import java.util.List;

public class FeatureScore implements CalculateScore {
    public static final int STAGNANT_STEPS = 500;
    public static final int MINI_BATCH_SIZE = 50;
    public static final double LEARNING_RATE = 1e-2;
    public static final double L1 = 0;
    public static final double L2 = 1e-8;
    private final Population population;
    private final BasicNetwork network;
    private MLDataSet trainingData;
    private MLDataSet validationData;
    private boolean init;
    private int hiddenCount;
    private int maxIterations;
    private boolean shouldNeuralReport = false;
    private double bestValidationError;


    public FeatureScore(MLDataSet theTrainingData, MLDataSet theValidationData, Population thePopulation, int theHiddenCount, int theMaxIterations) {
        this.trainingData = theTrainingData;
        this.validationData = theValidationData;
        this.population = thePopulation;
        this.network = new BasicNetwork();
        this.hiddenCount = theHiddenCount;
        this.maxIterations = theMaxIterations;
        network.addLayer(new BasicLayer(null,true,this.population.getPopulationSize()));
        network.addLayer(new BasicLayer(new ActivationReLU(),true,this.hiddenCount));
        network.addLayer(new BasicLayer(new ActivationLinear(),false,this.trainingData.getIdealSize()));
        network.getStructure().finalizeStructure();
    }

    private void randomizeNetwork() {
        (new XaiverRandomizer(41)).randomize(network);
    }

    public static void cleanVector(MLData vec) {
        for(int i = 0; i<vec.size(); i++ ) {
            double d = vec.getData(i);
            if( Double.isInfinite(d) || Double.isNaN(d) ) {
                vec.setData(i, 0);
            }
        }
    }

    private MLDataSet encodeDataset(List<Genome> genomes, MLDataSet dataset) {
        int inputSize = this.network.getInputCount();
        MLDataSet engineeredDataset = new BasicMLDataSet();

        for(MLDataPair pair: dataset) {
            MLData engineeredInput = new BasicMLData(inputSize);
            MLData engineeredIdeal = new BasicMLData(this.network.getOutputCount());
            MLDataPair engineeredPair = new BasicMLDataPair(engineeredInput,engineeredIdeal);

            // Copy ideal
            for(int i=0;i<pair.getIdeal().size();i++){
                engineeredIdeal.setData(i, pair.getIdeal().getData(i));
            }

            // Create input
            for(int i=0;i<inputSize;i++) {
                double d = 0.0;

                if( i< genomes.size() ) {
                    MLRegression phen = (MLRegression) genomes.get(i);
                    try {
                        MLData output = phen.compute(pair.getInput());
                        d = output.getData(0);
                    } catch (EARuntimeError ex) {
                        d = 0.0;
                    }
                }
                engineeredInput.setData(i, d);
            }

            cleanVector(engineeredPair.getInput());

            // add to dataset
            engineeredDataset.add(engineeredPair);
        }

        Transform.zscore(engineeredDataset);
        return engineeredDataset;
    }

    private void reportNeuralTrain(MLTrain train, EarlyStoppingStrategy earlyStop) {
        if( this.shouldNeuralReport ) {
            System.out.println("Epoch #" + train.getIteration() + " Train Error:" + Format.formatDouble(train.getError(), 6)
                    + ", Validation Error: " + Format.formatDouble(earlyStop.getValidationError(), 6) +
                    ", Stagnant: " + earlyStop.getStagnantIterations());
        }
    }

    public void calculateScores() {
        List<Genome> genomes = this.population.flatten();

        // Create a new training set, with the new engineered autofeatures
        MLDataSet engineeredTrainingSet = encodeDataset(genomes, this.trainingData);
        MLDataSet engineeredValidationSet = encodeDataset(genomes, this.trainingData);

        boolean done = false;

        while(!done) {
            randomizeNetwork();

            // Train a neural network with engineered dataset
            StochasticGradientDescent train = new StochasticGradientDescent(network, engineeredTrainingSet);
            train.setUpdateRule(new AdamUpdate());
            train.setBatchSize(MINI_BATCH_SIZE);
            train.setL1(L1);
            train.setL2(L2);
            train.setLearningRate(LEARNING_RATE);


            //final Backpropagation train = new Backpropagation(network, engineeredTrainingSet, learningRate, this.momentum);

            //final ResilientPropagation train = new ResilientPropagation(network, engineeredDataset);
            //train.setErrorFunction(new CrossEntropyErrorFunction());
            //train.setNesterovUpdate(true);

            EarlyStoppingStrategy earlyStop = new EarlyStoppingStrategy(engineeredValidationSet,5,STAGNANT_STEPS);
            train.addStrategy(earlyStop);
            if( maxIterations>0 ) {
                train.addStrategy(new EndIterationsStrategy(maxIterations));
            }

            do {
                train.iteration();
                if( (train.getIteration()%500) == 0) {
                    reportNeuralTrain(train,earlyStop);
                }
            }
            while (!train.isTrainingDone() && !Double.isInfinite(train.getError()) && !Double.isNaN(train.getError()) );
            train.finishTraining();
            this.bestValidationError = earlyStop.getBestValidationError();

            if( !Double.isInfinite(train.getError()) && !Double.isNaN(train.getError()) ) {
                done = true;
                reportNeuralTrain(train,earlyStop);
            }
        }

        // Evaluate feature importance

        PerturbationFeatureImportanceCalc fi = new PerturbationFeatureImportanceCalc();
        fi.init(network,null);
        fi.performRanking(engineeredValidationSet);

        int count = Math.min(fi.getFeatures().size(),genomes.size());// might not be needed
        for(int i=0;i<count;i++) {
            FeatureRank rank = fi.getFeatures().get(i);
            genomes.get(i).setScore(rank.getImportancePercent());
        }

        this.init = true;
    }

    @Override
    public double calculateScore(MLMethod method) {
        if(this.init) {
            Genome genome = (Genome)method;
            if( Double.isInfinite(genome.getScore()) || Double.isNaN(genome.getScore()) ) {
                return 0;
            }
            return ((Genome) method).getScore();
        } else {
            throw new EncogError("Must calculate scores first.");
        }
    }

    @Override
    public boolean shouldMinimize() {
        return true;
    }

    @Override
    public boolean requireSingleThreaded() {
        return false;
    }

    public double getBestValidationError() {
        return bestValidationError;
    }

    public void setBestValidationError(double bestValidationError) {
        this.bestValidationError = bestValidationError;
    }
}
