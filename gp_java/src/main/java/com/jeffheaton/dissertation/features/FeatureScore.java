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

    private void cleanVector(MLData vec) {
        for(int i = 0; i<vec.size(); i++ ) {
            double d = vec.getData(i);
            if( Double.isInfinite(d) || Double.isNaN(d) ) {
                vec.setData(i, 0);
            }
        }
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
                    try {
                        MLData output = phen.compute(pair.getInput());
                        engineeredInput.setData(networkIdx++,output.getData(0));
                    } catch(EARuntimeError ex) {
                        engineeredInput.setData(networkIdx++,0);
                    }

                }
            }
            cleanVector(engineeredPair.getInput());

            // add to dataset
            engineeredDataset.add(engineeredPair);
        }

        // Train a neural network with engineered dataset
        //final Backpropagation train = new Backpropagation(network, engineeredDataset, 1e-30, 0.9);
        //NeuralPSO train = new NeuralPSO(network,engineeredDataset);

        final ResilientPropagation train = new ResilientPropagation(network, engineeredDataset);
        train.setErrorFunction(new CrossEntropyErrorFunction());
        //train.setNesterovUpdate(true);

        int epoch = 1;
        int stalled = 0;

        double bestError = Double.POSITIVE_INFINITY;
        do {
            train.iteration();
            if( train.getError()<bestError) {
                stalled = 0;
                bestError = train.getError();
            } else {
                stalled++;
            }
            System.out.println("Epoch #" + epoch + " Error:" + train.getError());
            epoch++;
        } while( epoch<1000 && stalled<10 && !Double.isInfinite(train.getError()) && !Double.isInfinite(train.getError()));
        train.finishTraining();

        // Evaluate feature importance

        NeuralFeatureImportanceCalc fi = new NeuralFeatureImportanceCalc(network);
        fi.calculateFeatureImportance();

        for(int i=0;i<fi.getFeatures().size();i++) {
            FeatureRanking rank = fi.getFeatures().get(i);
            genomes.get(i).setScore(1.0-rank.getImportancePercent());
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
