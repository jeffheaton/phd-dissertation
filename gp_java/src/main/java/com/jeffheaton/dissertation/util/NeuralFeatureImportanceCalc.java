package com.jeffheaton.dissertation.util;

import org.encog.EncogError;
import org.encog.neural.networks.BasicNetwork;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * Created by Jeff on 3/31/2016.
 */
public class NeuralFeatureImportanceCalc {
    private final List<FeatureRanking> features = new ArrayList<>();
    private final BasicNetwork network;

    public NeuralFeatureImportanceCalc(BasicNetwork theNetwork) {
        this.network = theNetwork;


        for (int i = 0; i < theNetwork.getInputCount(); i++) {
            this.features.add(new FeatureRanking("f" + i));
        }
    }

    public NeuralFeatureImportanceCalc(BasicNetwork theNetwork, String[] theFeatureNames) {
        this.network = theNetwork;

        if (theFeatureNames.length != network.getInputCount()) {
            throw new EncogError("Neural network inputs(" + theNetwork.getInputCount() + ") and feature name count("
                    + theFeatureNames.length + ") do not match.");
        }

        for (String name : theFeatureNames) {
            this.features.add(new FeatureRanking(name));
        }
    }

    public void calculateFeatureImportance() {
        reset();

        // Sum weights for each input neuron
        for (int inputNueron = 0; inputNueron < this.network.getInputCount(); inputNueron++) {
            FeatureRanking ranking = this.features.get(inputNueron);
            for (int nextNeuron = 0; nextNeuron < this.network.getLayerNeuronCount(1); nextNeuron++) {
                double i_h = network.getWeight(0, inputNueron, nextNeuron);
                double h_o = network.getWeight(1, nextNeuron, 0);
                ranking.addWeight(i_h * h_o);
            }

        }
        // sum total weight to input neurons.
        double sum = 0;
        for (FeatureRanking rank : this.features) {
            sum += Math.abs(rank.getTotalWeight());
        }

        // calculate each feature's importance percent
        for (FeatureRanking rank : this.features) {
            rank.setImportancePercent(Math.abs(rank.getTotalWeight()) / sum);
        }

        this.features.sort(new Comparator<FeatureRanking>() {
            @Override
            public int compare(FeatureRanking o1, FeatureRanking o2) {
                return Double.compare(o2.getImportancePercent(), o1.getImportancePercent());
            }
        });
    }

    public void reset() {
        for (FeatureRanking rank : this.features) {
            rank.setImportancePercent(0);
            rank.setTotalWeight(0);
        }
    }

    @Override
    public String toString() {
        return this.features.toString();
    }

    public List<FeatureRanking> getFeatures() {
        return features;
    }

    public double calculateDeviation() {
        double sum = 0;
        for (FeatureRanking rank : this.features) {
            sum += rank.getImportancePercent();
        }
        double mean = sum / this.features.size();

        sum = 0;
        for (FeatureRanking rank : this.features) {
            double d = mean - rank.getImportancePercent();
            sum += d * d;
        }
        return Math.sqrt(sum / this.features.size());
    }
}
