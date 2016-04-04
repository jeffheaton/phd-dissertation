package com.jeffheaton.dissertation.features;

import com.jeffheaton.dissertation.util.FeatureRanking;
import com.jeffheaton.dissertation.util.NeuralFeatureImportanceCalc;
import com.jeffheaton.dissertation.util.NewSimpleEarlyStoppingStrategy;
import org.encog.EncogError;
import org.encog.engine.network.activation.ActivationLinear;
import org.encog.engine.network.activation.ActivationReLU;
import org.encog.mathutil.error.ErrorCalculation;
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
import org.encog.ml.ea.codec.GeneticCODEC;
import org.encog.ml.ea.exception.EARuntimeError;
import org.encog.ml.ea.genome.Genome;
import org.encog.ml.ea.population.Population;
import org.encog.ml.ea.species.Species;
import org.encog.neural.error.CrossEntropyErrorFunction;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.back.Backpropagation;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;
import org.encog.neural.networks.training.pso.NeuralPSO;
import org.encog.util.Format;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by Jeff on 3/31/2016.
 */
public class FeatureScore implements CalculateScore {

    private final Population population;
    private final BasicNetwork network;
    private MLDataSet trainingData;
    private MLDataSet validationData;
    private boolean init;
    private int maxEpoch = 1000;
    private int maxStagnant = 10;
    private int hiddenCount = 500;
    private double learningRate = 1e-9;
    private double momentum = 0.9;

    public FeatureScore(MLDataSet theTrainingData, MLDataSet theValidationData, Population thePopulation) {
        this.trainingData = theTrainingData;
        this.validationData = theValidationData;
        this.population = thePopulation;
        this.network = new BasicNetwork();
        network.addLayer(new BasicLayer(null,true,this.population.getPopulationSize()));
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
        MLDataSet engineeredDataset = new BasicMLDataSet();

        for(MLDataPair pair: dataset) {
            int networkIdx = 0;

            MLData engineeredInput = new BasicMLData(this.network.getInputCount());
            MLData engineeredIdeal = new BasicMLData(this.network.getOutputCount());
            MLDataPair engineeredPair = new BasicMLDataPair(engineeredInput,engineeredIdeal);

            // Copy ideal
            for(int i=0;i<pair.getIdeal().size();i++){
                engineeredIdeal.setData(i, pair.getIdeal().getData(i));
            }

            // Create input
            for(Genome genome: genomes) {
                MLRegression phen = (MLRegression) genome;
                try {
                    MLData output = phen.compute(pair.getInput());
                    engineeredInput.setData(networkIdx++,output.getData(0));
                } catch(EARuntimeError ex) {
                    engineeredInput.setData(networkIdx++,0);
                }

            }

            cleanVector(engineeredPair.getInput());

            // add to dataset
            engineeredDataset.add(engineeredPair);
        }
        return engineeredDataset;
    }

    public void calculateScores() {
        List<Genome> genomes = this.population.flatten();
        randomizeNetwork();

        // Create a new training set, with the new engineered features
        MLDataSet engineeredTrainingSet = encodeDataset(genomes, this.trainingData);
        MLDataSet engineeredValidationSet = encodeDataset(genomes, this.trainingData);

        // Train a neural network with engineered dataset
        final Backpropagation train = new Backpropagation(network, engineeredTrainingSet, this.learningRate, this.momentum);
        //NeuralPSO train = new NeuralPSO(network,engineeredDataset);

        //final ResilientPropagation train = new ResilientPropagation(network, engineeredDataset);
        train.setErrorFunction(new CrossEntropyErrorFunction());
        train.setNesterovUpdate(true);

        int epoch = 1;

        NewSimpleEarlyStoppingStrategy earlyStop = new NewSimpleEarlyStoppingStrategy(engineeredValidationSet);
        train.addStrategy(earlyStop);

        double bestError = Double.POSITIVE_INFINITY;
        do {
            train.iteration();
            System.out.println("Epoch #" + epoch + " Train Error:" + Format.formatDouble(train.getError(),6)
                    + ", Validation Error: " + Format.formatDouble(earlyStop.getValidationError(),6) +
                    ", Stagnant: " + earlyStop.getStagnantIterations());
            epoch++;
        } while( !train.isTrainingDone() && !Double.isInfinite(train.getError()) && !Double.isInfinite(train.getError()));
        train.finishTraining();

        // Evaluate feature importance

        NeuralFeatureImportanceCalc fi = new NeuralFeatureImportanceCalc(network);
        fi.calculateFeatureImportance();

        int count = Math.min(fi.getFeatures().size(),genomes.size());// might not be needed
        for(int i=0;i<count;i++) {
            FeatureRanking rank = fi.getFeatures().get(i);
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
        return false;
    }

    @Override
    public boolean requireSingleThreaded() {
        return false;
    }
}
