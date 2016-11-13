package com.jeffheaton.dissertation.autofeatures;

import com.jeffheaton.dissertation.JeffDissertation;
import org.encog.EncogError;
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
import org.encog.util.EngineArray;
import org.encog.util.Format;

import java.util.List;

public class FeatureScore implements CalculateScore {

    public static int MAX_RETRY_STABLE = 5;
    private final Population population;
    private final BasicNetwork network;
    private MLDataSet trainingData;
    private boolean init;
    private int maxIterations;
    private double bestValidationError;
    private AutoEngineerFeatures owner;
    private MLDataSet engineeredDataset;

    public FeatureScore(AutoEngineerFeatures theOwner, MLDataSet theTrainingData, Population thePopulation, int theMaxIterations) {
        this.owner = theOwner;
        this.trainingData = theTrainingData;
        this.population = thePopulation;
        this.network = JeffDissertation.factorNeuralNetwork(
                this.population.getPopulationSize(),
                this.trainingData.getIdealSize(),
                true);
        this.maxIterations = theMaxIterations;
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

    private void encodeDataset(List<Genome> genomes) {
        int inputSize = this.network.getInputCount();

        for(int i=0;i<this.owner.getRankingSet().size();i++) {
            MLDataPair pair = this.owner.getRankingSet().get(i);
            MLDataPair engineeredPair = this.engineeredDataset.get(i);

            // Copy ideal
            EngineArray.arrayCopy(pair.getIdeal().getData(),engineeredPair.getIdeal().getData());

            // Create input
            MLData engineeredInput = engineeredPair.getInput();
            for(int inputIdx=0;inputIdx<inputSize;inputIdx++) {
                double d = 0.0;

                if( inputIdx< genomes.size() ) {
                    MLRegression phen = (MLRegression) genomes.get(inputIdx);
                    try {
                        MLData output = phen.compute(pair.getInput());
                        d = output.getData(0);
                    } catch (EARuntimeError ex) {
                        d = 0.0;
                    }
                }
                engineeredInput.setData(inputIdx, d);
            }

            cleanVector(engineeredPair.getInput());
        }

        Transform.zscore(engineeredDataset);
    }

    private void reportNeuralTrain(MLTrain train) {
        if( this.owner.isShouldReportNeural() ) {
            System.out.println("Epoch #" + train.getIteration() + " Train Error:" + Format.formatDouble(train.getError(), 6));
        }
    }

    private void initEngineeredDataset() {
        // create dataset to hold engineered features
        this.engineeredDataset = new BasicMLDataSet();
        int inputSize = this.network.getInputCount();

        for(MLDataPair pair: this.owner.getRankingSet()) {
            MLData engineeredInput = new BasicMLData(inputSize);
            MLData engineeredIdeal = new BasicMLData(this.network.getOutputCount());
            MLDataPair engineeredPair = new BasicMLDataPair(engineeredInput, engineeredIdeal);
            this.engineeredDataset.add(engineeredPair);
        }
    }

    public boolean calculateScores() {

        if( this.engineeredDataset == null) {
            initEngineeredDataset();
        }

        List<Genome> genomes = this.population.flatten();

        // Create a new training set, with the new engineered autofeatures
        encodeDataset(genomes);

        boolean done = false;
        int unstableTry = 0;

        while(!done) {
            randomizeNetwork();

            // Train a neural network with engineered dataset
            JeffDissertation.DissertationNeuralTraining d = JeffDissertation.factorNeuralTrainer(
                    network,this.engineeredDataset,null);
            MLTrain train = d.getTrain();
            EarlyStoppingStrategy earlyStop = d.getEarlyStop();

            if( maxIterations>0 ) {
                train.addStrategy(new EndIterationsStrategy(maxIterations));
            }

            train.addStrategy(new EndIterationsStrategy(maxIterations));

            do {
                train.iteration();
                reportNeuralTrain(train);
            }
            while (!train.isTrainingDone() && !Double.isInfinite(train.getError()) && !Double.isNaN(train.getError()) );
            train.finishTraining();
            this.bestValidationError = train.getError();

            if( !Double.isInfinite(train.getError()) && !Double.isNaN(train.getError()) ) {
                done = true;
                reportNeuralTrain(train);
            } else {
                this.owner.report("Network became unstable, try: " + unstableTry);

                if( unstableTry>= FeatureScore.MAX_RETRY_STABLE ) {
                    return false;
                }
                unstableTry++;
            }
        }

        // Evaluate feature importance

        PerturbationFeatureImportanceCalc fi = new PerturbationFeatureImportanceCalc();
        fi.init(network,null);
        fi.performRanking(this.engineeredDataset);

        int count = Math.min(fi.getFeatures().size(),genomes.size());// might not be needed
        for(int i=0;i<count;i++) {
            FeatureRank rank = fi.getFeatures().get(i);
            genomes.get(i).setScore(rank.getImportancePercent());
        }

        this.init = true;
        return true;
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
