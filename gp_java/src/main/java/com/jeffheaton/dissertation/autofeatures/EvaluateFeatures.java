package com.jeffheaton.dissertation.autofeatures;

import org.encog.Encog;
import org.encog.EncogError;
import org.encog.engine.network.activation.ActivationLinear;
import org.encog.engine.network.activation.ActivationReLU;
import org.encog.mathutil.error.ErrorCalculation;
import org.encog.mathutil.error.ErrorCalculationMode;
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
import org.encog.util.csv.CSVFormat;
import org.encog.util.simple.EncogUtility;

import java.io.File;
import java.util.List;

public class EvaluateFeatures {
    public static final int STAGNANT_STEPS = 500;
    public static final int MINI_BATCH_SIZE = 32;
    public static final double LEARNING_RATE = 1e-2;
    public static final double L1 = 0;
    public static final double L2 = 1e-8;
    private final BasicNetwork network;
    private MLDataSet trainingData;
    private boolean init;
    private int hiddenCount = 500;
    private boolean shouldNeuralReport = true;
    private AutoFeatureGP owner;

    public EvaluateFeatures(MLDataSet theTrainingData, AutoFeatureGP theOwner) {
        this.owner = theOwner;
        this.trainingData = theTrainingData;
        this.network = new BasicNetwork();
        network.addLayer(new BasicLayer(null,true,this.owner.getPopulation().getPopulationSize()));
        network.addLayer(new BasicLayer(new ActivationReLU(),true,this.hiddenCount));
        network.addLayer(new BasicLayer(new ActivationLinear(),false,this.trainingData.getIdealSize()));
        network.getStructure().finalizeStructure();
    }

    private void randomizeNetwork() {
        (new XaiverRandomizer(41)).randomize(network);
    }

    private void cleanVector(MLData vec) {
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

        Transform.interpolate(engineeredDataset);
        Transform.zscore(engineeredDataset);

        return engineeredDataset;
    }

    private void reportNeuralTrain(MLTrain train) {
        if( this.shouldNeuralReport ) {
            System.out.println("Epoch #" + train.getIteration() + " Train Error:" + Format.formatDouble(train.getError(), 6));
        }
    }

    public double calculateScores() {
        double error = Double.NaN;

        List<Genome> genomes = this.owner.getPopulation().flatten();

        // Create a new training set, with the new engineered autofeatures
        MLDataSet engineeredTrainingSet = encodeDataset(genomes, this.trainingData);

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
            train.addStrategy(new EndIterationsStrategy(1000));



            do {
                train.iteration();
                reportNeuralTrain(train);
            }
            while (!train.isTrainingDone() && !Double.isInfinite(train.getError()) && !Double.isNaN(train.getError()) );

            EncogUtility.saveCSV(new File("c:\\test\\out.csv"), CSVFormat.EG_FORMAT,engineeredTrainingSet);
            ErrorCalculation.setMode(ErrorCalculationMode.RMS);

            error = EncogUtility.calculateRegressionError(network,engineeredTrainingSet);
            train.finishTraining();

            if( !Double.isInfinite(train.getError()) && !Double.isNaN(train.getError()) ) {
                done = true;
                reportNeuralTrain(train);
            }
        }

        // Evaluate feature importance

        PerturbationFeatureImportanceCalc fi = new PerturbationFeatureImportanceCalc();
        fi.init(network,null);
        fi.performRanking(engineeredTrainingSet);

        int count = Math.min(fi.getFeatures().size(),genomes.size());// might not be needed
        for(int i=0;i<count;i++) {
            FeatureRank rank = fi.getFeatures().get(i);
            genomes.get(i).setScore(rank.getImportancePercent());
        }

        this.init = true;
        return error;
    }
}
