package com.jeffheaton.dissertation.experiments;

import com.jeffheaton.dissertation.data.AutoMPG;
import com.jeffheaton.dissertation.util.FeatureRanking;
import com.jeffheaton.dissertation.util.NeuralFeatureImportanceCalc;
import com.jeffheaton.dissertation.util.NewSimpleEarlyStoppingStrategy;
import com.jeffheaton.dissertation.util.Transform;
import org.encog.Encog;
import org.encog.engine.network.activation.ActivationLinear;
import org.encog.engine.network.activation.ActivationReLU;
import org.encog.mathutil.error.ErrorCalculation;
import org.encog.mathutil.error.ErrorCalculationMode;
import org.encog.mathutil.randomize.XaiverRandomizer;
import org.encog.mathutil.randomize.generate.MersenneTwisterGenerateRandom;
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
import org.encog.util.Format;
import org.encog.util.csv.CSVFormat;
import org.encog.util.csv.ReadCSV;

import java.io.InputStream;
import java.util.Random;


public class ExperimentNeuralAutoMPG {

    public void runNeural() {
        ErrorCalculation.setMode(ErrorCalculationMode.RMS);

        MLDataSet dataset = AutoMPG.getInstance().loadData();
        Transform.zscore(dataset);

        // split
        MLDataSet[] split = Transform.splitTrainValidate(dataset,new MersenneTwisterGenerateRandom(42),0.75);
        MLDataSet trainingSet = split[0];
        MLDataSet validationSet = split[1];

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
        final Backpropagation train = new Backpropagation(network, trainingSet, 1e-5, 0.9);
        //final ResilientPropagation train = new ResilientPropagation(network, trainingSet);
        train.setErrorFunction(new CrossEntropyErrorFunction());
        train.setNesterovUpdate(true);
        NewSimpleEarlyStoppingStrategy earlyStop = new NewSimpleEarlyStoppingStrategy(validationSet);
        train.addStrategy(earlyStop);

        int epoch = 1;

        do {
            train.iteration();
            System.out.println("Epoch #" + epoch + " Train Error:" + Format.formatDouble(train.getError(),6)
                    + ", Validation Error: " + Format.formatDouble(earlyStop.getValidationError(),6) +
                     ", Stagnant: " + earlyStop.getStagnantIterations());

            epoch++;
        } while(!train.isTrainingDone());
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

        MLDataSet dataset = AutoMPG.getInstance().loadData();

        // split
        MLDataSet[] split = Transform.splitTrainValidate(dataset,new MersenneTwisterGenerateRandom(42),0.75);
        MLDataSet trainingSet = split[0];
        MLDataSet validationSet = split[1];

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
        genetic.addScoreAdjuster(new ComplexityAdjustedScore(10,20,10,50.0));
        genetic.getRules().addRewriteRule(new RewriteConstants());
        genetic.getRules().addRewriteRule(new RewriteAlgebraic());
        genetic.setSpeciation(new PrgSpeciation());

        NewSimpleEarlyStoppingStrategy earlyStop = new NewSimpleEarlyStoppingStrategy(validationSet);
        genetic.addStrategy(earlyStop);

        (new RampedHalfAndHalf(context,1, 6)).generate(new Random(), pop);

        genetic.setShouldIgnoreExceptions(false);

        EncogProgram best = null;

        try {

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
        prg.runGP();
    }
}
