package com.jeffheaton.dissertation.experiments;

import com.jeffheaton.dissertation.util.FeatureRanking;
import com.jeffheaton.dissertation.util.NeuralFeatureImportanceCalc;
import com.jeffheaton.dissertation.util.Transform;
import org.encog.Encog;
import org.encog.engine.network.activation.ActivationLinear;
import org.encog.engine.network.activation.ActivationReLU;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.mathutil.error.ErrorCalculation;
import org.encog.mathutil.error.ErrorCalculationMode;
import org.encog.mathutil.randomize.XaiverRandomizer;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataPair;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.ml.ea.score.adjust.ComplexityAdjustedScore;
import org.encog.ml.ea.train.basic.TrainEA;
import org.encog.ml.fitness.MultiObjectiveFitness;
import org.encog.ml.prg.EncogProgram;
import org.encog.ml.prg.EncogProgramContext;
import org.encog.ml.prg.PrgCODEC;
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
import org.encog.neural.networks.training.propagation.back.Backpropagation;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;
import org.encog.util.csv.CSVFormat;
import org.encog.util.csv.ReadCSV;

import java.io.File;
import java.io.InputStream;
import java.util.Random;


public class ExperimentNeuralAutoMPG {

    public static MLDataSet loadCSV(ReadCSV csv, int[] input, int[] ideal) {
        MLDataSet result = new BasicMLDataSet();
        while(csv.next()) {
            MLData inputItem = new BasicMLData(input.length);
            MLData idealItem = new BasicMLData(ideal.length);
            MLDataPair pair = new BasicMLDataPair(inputItem,idealItem);

            // Read input
            int idx = 0;
            for(int i:input) {
                inputItem.setData(idx++,csv.getDouble(i));
            }

            // Read ideal
            idx = 0;
            for(int i:ideal) {
                idealItem.setData(idx++,csv.getDouble(i));
            }

            result.add(pair);

        }
        return result;
    }

    public InputStream loadDatasetMPG() {
        final InputStream istream = this.getClass().getResourceAsStream("/auto-mpg.csv");
        if (istream == null) {
            System.out.println("Cannot access data set, make sure the resources are available.");
            System.exit(1);
        }
        return istream;
    }

    public void seedInput(BasicNetwork network) {
        for(int i=0;i<network.getInputCount();i++) {
            for(int j=0;j<network.getLayerNeuronCount(1);j++) {
                network.setWeight(0,i,j,0.5);
            }
        }
    }

    public void runNeural() {
        ErrorCalculation.setMode(ErrorCalculationMode.RMS);

        InputStream is = loadDatasetMPG();
        ReadCSV csv = new ReadCSV(is,true,CSVFormat.EG_FORMAT.DECIMAL_POINT);
        MLDataSet trainingSet = loadCSV(csv,new int[] {1,2,3,4,5,6,7}, new int[] {0});
        Transform.interpolate(trainingSet);
        Transform.zscore(trainingSet);

        // create a neural network, without using a factory
        BasicNetwork network = new BasicNetwork();
        network.addLayer(new BasicLayer(null,true,trainingSet.getInputSize()));
        network.addLayer(new BasicLayer(new ActivationReLU(),true,500));
        //network.addLayer(new BasicLayer(new ActivationReLU(),true,50));
        //network.addLayer(new BasicLayer(new ActivationReLU(),true,15));
        network.addLayer(new BasicLayer(new ActivationLinear(),false,trainingSet.getIdealSize()));
        network.getStructure().finalizeStructure();
        (new XaiverRandomizer()).randomize(network);
        //seedInput(network);

        // train the neural network
        final Backpropagation train = new Backpropagation(network, trainingSet, 1e-6, 0.9);
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
        } while(train.getError() > 2.5);
        train.finishTraining();


        NeuralFeatureImportanceCalc fi = new NeuralFeatureImportanceCalc(network,new String[] {
                "cylinders",
                "displacement",
                "horsepower",
                "weight",
                "acceleration",
                "year",
                "origin"
        });
        fi.calculateFeatureImportance();
        for (FeatureRanking ranking : fi.getFeatures()) {
            System.out.println(ranking.toString());
        }
        Encog.getInstance().shutdown();


    }

    public void runGP() {
        ErrorCalculation.setMode(ErrorCalculationMode.RMS);

        InputStream is = loadDatasetMPG();
        ReadCSV csv = new ReadCSV(is,true,CSVFormat.EG_FORMAT.DECIMAL_POINT);
        MLDataSet trainingSet = loadCSV(csv,new int[] {1,2,3,4,5,6,7}, new int[] {0});
        Transform.interpolate(trainingSet);

        EncogProgramContext context = new EncogProgramContext();
        context.defineVariable("c");
        context.defineVariable("d");
        context.defineVariable("h");
        context.defineVariable("w");
        context.defineVariable("a");
        context.defineVariable("y");
        context.defineVariable("o");

        StandardExtensions.createNumericOperators(context);

        PrgPopulation pop = new PrgPopulation(context,1000);

        MultiObjectiveFitness score = new MultiObjectiveFitness();
        score.addObjective(1.0, new TrainingSetScore(trainingSet));

        TrainEA genetic = new TrainEA(pop, score);
        //genetic.setValidationMode(true);
        genetic.setCODEC(new PrgCODEC());
        genetic.addOperation(0.5, new SubtreeCrossover());
        genetic.addOperation(0.25, new ConstMutation(context,0.5,1.0));
        genetic.addOperation(0.25, new SubtreeMutation(context,4));
        genetic.addScoreAdjuster(new ComplexityAdjustedScore(10,20,10,20.0));
        genetic.getRules().addRewriteRule(new RewriteConstants());
        genetic.getRules().addRewriteRule(new RewriteAlgebraic());
        genetic.setSpeciation(new PrgSpeciation());

        (new RampedHalfAndHalf(context,1, 6)).generate(new Random(), pop);

        genetic.setShouldIgnoreExceptions(false);

        EncogProgram best = null;

        try {

            for (int i = 0; i < 100000; i++) {
                genetic.iteration();
                best = (EncogProgram) genetic.getBestGenome();
                System.out.println(genetic.getIteration() + ", Error: "
                        + best.getScore() + ",Best Genome Size:" +best.size()
                        + ",Species Count:" + pop.getSpecies().size() + ",best: " + best.dumpAsCommonExpression());
            }

            //EncogUtility.evaluate(best, trainingData);

            System.out.println("Final score:" + best.getScore()
                    + ", effective score:" + best.getAdjustedScore());
            System.out.println(best.dumpAsCommonExpression());
            //pop.dumpMembers(Integer.MAX_VALUE);
            //pop.dumpMembers(10);

        } catch (Throwable t) {
            t.printStackTrace();
        } finally {
            genetic.finishTraining();
            Encog.getInstance().shutdown();
        }
    }


    public static void main(String[] args) {
        ExperimentNeuralAutoMPG prg = new ExperimentNeuralAutoMPG();
        prg.runNeural();
    }
}
