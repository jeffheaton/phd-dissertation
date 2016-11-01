package com.jeffheaton.dissertation;

import com.jeffheaton.dissertation.experiments.*;
import com.jeffheaton.dissertation.experiments.misc.ExperimentAutoFeature;
import com.jeffheaton.dissertation.experiments.misc.ExperimentGPFile;
import com.jeffheaton.dissertation.experiments.misc.ExperimentSimpleGP;
import com.jeffheaton.dissertation.util.SimpleGPConstraint;
import org.encog.engine.network.activation.ActivationLinear;
import org.encog.engine.network.activation.ActivationReLU;
import org.encog.engine.network.activation.ActivationSoftMax;
import org.encog.mathutil.randomize.XaiverRandomizer;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.ea.population.Population;
import org.encog.ml.ea.score.adjust.ComplexityAdjustedScore;
import org.encog.ml.ea.train.basic.TrainEA;
import org.encog.ml.fitness.MultiObjectiveFitness;
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
import org.encog.ml.train.strategy.end.EarlyStoppingStrategy;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.TrainingSetScore;
import org.encog.neural.networks.training.propagation.sgd.StochasticGradientDescent;
import org.encog.neural.networks.training.propagation.sgd.update.AdamUpdate;
import org.encog.neural.networks.training.propagation.sgd.update.UpdateRule;

import java.util.Random;

public class JeffDissertation {

    public static final int NEURAL_REPEAT_COUNT = 5;
    public static final int GENETIC_REPEAT_COUNT = 5;
    public static final long RANDOM_SEED = 42;
    public static final double LEARNING_RATE = 1e-2;
    public static final int STAGNANT_NEURAL = 100;
    public static final int STAGNANT_GENETIC = 100;
    public static final double L1 = 0;
    public static final double L2 = 1e-8;
    public static final double TRAIN_VALIDATION_SPLIT = 0.75;
    public static final int MINI_BATCH_SIZE = 32;
    public static final int POPULATION_SIZE = 100;
    public static final Class UPDATE_RULE = AdamUpdate.class;
    public static final double MINIMUM_IMPROVE = 0.01;
    public static final int EXP4_PATTERN_COUNT = 25;

    public static class DissertationNeuralTraining {
        private final EarlyStoppingStrategy earlyStop;
        private final MLTrain train;

        public DissertationNeuralTraining(MLTrain train, EarlyStoppingStrategy earlyStop) {
            this.earlyStop = earlyStop;
            this.train = train;
        }

        public EarlyStoppingStrategy getEarlyStop() {
            return earlyStop;
        }

        public MLTrain getTrain() {
            return train;
        }
    }

    public static class DissertationGeneticTraining {
        private final PrgPopulation population;
        private final EarlyStoppingStrategy earlyStop;
        private final TrainEA train;

        public DissertationGeneticTraining(PrgPopulation population, EarlyStoppingStrategy earlyStop, TrainEA train) {
            this.population = population;
            this.earlyStop = earlyStop;
            this.train = train;
        }

        public PrgPopulation getPopulation() {
            return population;
        }

        public EarlyStoppingStrategy getEarlyStop() {
            return earlyStop;
        }

        public TrainEA getTrain() {
            return train;
        }
    }


    public static UpdateRule factorUpdateRule() {
        try {
            return (UpdateRule)UPDATE_RULE.newInstance();
        } catch (InstantiationException e) {
            return null;
        } catch (IllegalAccessException e) {
            return null;
        }
    }

    public static BasicNetwork factorNeuralNetwork(int inputCount, int outputCount, boolean regression) {
        BasicNetwork network = new BasicNetwork();
        network.addLayer(new BasicLayer(null, true, inputCount));
        int hiddenCount = inputCount * 2;

        network.addLayer(new BasicLayer(new ActivationReLU(), true, hiddenCount));
        hiddenCount = Math.max(2, hiddenCount/2);
        network.addLayer(new BasicLayer(new ActivationReLU(), true, hiddenCount));
        hiddenCount = Math.max(2, hiddenCount/2);
        network.addLayer(new BasicLayer(new ActivationReLU(), true, hiddenCount));
        hiddenCount = Math.max(2, hiddenCount/2);
        network.addLayer(new BasicLayer(new ActivationReLU(), true, hiddenCount));

        if (regression) {
            network.addLayer(new BasicLayer(new ActivationLinear(), false, outputCount));
        } else {
            network.addLayer(new BasicLayer(new ActivationSoftMax(), false, outputCount));
        }
        network.getStructure().finalizeStructure();
        (new XaiverRandomizer()).randomize(network);
        return network;
    }

    public static DissertationNeuralTraining factorNeuralTrainer(BasicNetwork network, MLDataSet trainingSet, MLDataSet validationSet) {
        StochasticGradientDescent train = new StochasticGradientDescent(network, trainingSet);
        train.setUpdateRule(JeffDissertation.factorUpdateRule());
        train.setBatchSize(JeffDissertation.MINI_BATCH_SIZE);
        train.setL1(JeffDissertation.L1);
        train.setL2(JeffDissertation.L2);
        train.setLearningRate(JeffDissertation.LEARNING_RATE);

        EarlyStoppingStrategy earlyStop = null;

        if( validationSet != null) {
            earlyStop = new EarlyStoppingStrategy(validationSet, 5, JeffDissertation.STAGNANT_NEURAL);
            earlyStop.setSaveBest(true);
            earlyStop.setMinimumImprovement(JeffDissertation.MINIMUM_IMPROVE);
            train.addStrategy(earlyStop);
        }

        return new DissertationNeuralTraining(train,earlyStop);
    }

    public static EncogProgramContext factorGeneticContext(String[] fields) {
        EncogProgramContext context = new EncogProgramContext();
        for (String field: fields) {
            context.defineVariable(field);
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

        return context;
    }

    public static DissertationGeneticTraining factorGeneticProgramming(
            EncogProgramContext context,
            MLDataSet trainingSet,
            MLDataSet validationSet,
            int populationSize) {
        PrgPopulation pop = new PrgPopulation(context, populationSize);

        MultiObjectiveFitness score = new MultiObjectiveFitness();
        score.addObjective(1.0, new TrainingSetScore(trainingSet));

        TrainEA genetic = new TrainEA(pop, score);
        genetic.setCODEC(new PrgCODEC());
        genetic.addOperation(0.5, new SubtreeCrossover());
        genetic.addOperation(0.25, new ConstMutation(context, 0.5, 1.0));
        genetic.addOperation(0.25, new SubtreeMutation(context, 4));
        genetic.addScoreAdjuster(new ComplexityAdjustedScore(10, 20, 50, 100.0));
        pop.getRules().addRewriteRule(new RewriteConstants());
        pop.getRules().addRewriteRule(new RewriteAlgebraic());
        pop.getRules().addConstraintRule(new SimpleGPConstraint());
        genetic.setSpeciation(new PrgSpeciation());
        genetic.setThreadCount(1);

        EarlyStoppingStrategy earlyStop = null;
        if( validationSet != null ) {
            earlyStop = new EarlyStoppingStrategy(validationSet, 5, JeffDissertation.STAGNANT_GENETIC);
            earlyStop.setMinimumImprovement(JeffDissertation.MINIMUM_IMPROVE);
            genetic.addStrategy(earlyStop);
        }

        (new RampedHalfAndHalf(context, 1, 6)).generate(new Random(), pop);

        genetic.setShouldIgnoreExceptions(false);
        return new JeffDissertation.DissertationGeneticTraining(pop,earlyStop,genetic);
    }


    public static void main(String[] args) {
        if( args[0].equalsIgnoreCase("simple-gp")) {
            (new ExperimentSimpleGP()).main(null);
        } else if( args[0].equalsIgnoreCase("feature-search")) {
            (new ExperimentAutoFeature()).main(null);
        } else if( args[0].equalsIgnoreCase("file-gp")) {
            (new ExperimentGPFile()).main(null);
        } else if( args[0].equalsIgnoreCase("experiment-1")) {
            (new PerformExperiment1()).main(null);
        } else if( args[0].equalsIgnoreCase("experiment-2")) {
            (new PerformExperiment2()).main(null);
        } else if( args[0].equalsIgnoreCase("experiment-3")) {
            (new PerformExperiment3()).main(null);
        } else if( args[0].equalsIgnoreCase("experiment-4")) {
            (new PerformExperiment4()).main(null);
        } else if( args[0].equalsIgnoreCase("experiment-5")) {
            (new PerformExperiment5()).main(null);
        } else if( args[0].equalsIgnoreCase("experiment-6")) {
            (new PerformExperiment6()).main(null);
        } else if( args[0].equalsIgnoreCase("experiment-1to5")) {
            (new PerformExperiments1To5()).main(null);
        }
    }
}
