package com.jeffheaton.dissertation.features;

import com.jeffheaton.dissertation.util.FeatureRanking;
import com.jeffheaton.dissertation.util.NeuralFeatureImportanceCalc;
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
import org.encog.ml.ea.genome.Genome;
import org.encog.ml.ea.population.Population;
import org.encog.ml.ea.species.Species;
import org.encog.neural.error.CrossEntropyErrorFunction;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.back.Backpropagation;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by Jeff on 3/31/2016.
 */
public class FeatureScore implements CalculateScore {

    private final MLDataSet dataset;
    private final Population population;
    private final BasicNetwork network;
    private boolean init;

    public FeatureScore(MLDataSet theDataset, Population thePopulation) {
        this.dataset = theDataset;
        this.population = thePopulation;
        this.network = new BasicNetwork();
        network.addLayer(new BasicLayer(null,true,this.population.getPopulationSize()));
        network.addLayer(new BasicLayer(new ActivationReLU(),true,500));
        network.addLayer(new BasicLayer(new ActivationLinear(),false,this.dataset.getIdealSize()));
        network.getStructure().finalizeStructure();
    }

    private void randomizeNetwork() {
        (new XaiverRandomizer(41)).randomize(network);
    }

    public void calculateScores() {
        List<Genome> genomes = new ArrayList<>();
        randomizeNetwork();

        // Create a new training set, with the new engineered features
        MLDataSet engineeredDataset = new BasicMLDataSet();


        for(MLDataPair pair: this.dataset) {
            int networkIdx = 0;

            MLData engineeredInput = new BasicMLData(this.network.getInputCount());
            MLData engineeredIdeal = new BasicMLData(this.network.getOutputCount());
            MLDataPair engineeredPair = new BasicMLDataPair(engineeredInput,engineeredIdeal);

            // Copy ideal
            for(int i=0;i<pair.getIdeal().size();i++){
                engineeredIdeal.setData(i, pair.getIdeal().getData(i));
            }

            // Create input
            for(Species species: this.population.getSpecies()) {
                for(Genome genome: species.getMembers()) {
                    genomes.add(genome);
                    MLRegression phen = (MLRegression) genome;
                    MLData output = phen.compute(pair.getInput());
                    engineeredInput.setData(networkIdx++,output.getData(0));
                }
            }

            // add to dataset
            engineeredDataset.add(engineeredPair);
        }

        // Train a neural network with engineered dataset
        final Backpropagation train = new Backpropagation(network, engineeredDataset, 1e-6, 0.9);
        //final ResilientPropagation train = new ResilientPropagation(network, trainingSet);
        train.setErrorFunction(new CrossEntropyErrorFunction());
        train.setNesterovUpdate(true);

        int epoch = 1;

        do {
            train.iteration();
            System.out.println("Epoch #" + epoch + " Error:" + train.getError());
            epoch++;
            if(epoch>100000) {
                break;
            }
        } while(train.getError() > 2.6);
        train.finishTraining();

        // Evaluate feature importance

        NeuralFeatureImportanceCalc fi = new NeuralFeatureImportanceCalc(network);
        fi.calculateFeatureImportance();

        for(int i=0;i<fi.getFeatures().size();i++) {
            FeatureRanking rank = fi.getFeatures().get(i);
            genomes.get(i).setScore(rank.getImportancePercent());
        }

        this.init = true;
    }

    @Override
    public double calculateScore(MLMethod method) {
        if(this.init) {
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
}
