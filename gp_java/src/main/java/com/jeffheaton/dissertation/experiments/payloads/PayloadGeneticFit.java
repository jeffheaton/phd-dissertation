package com.jeffheaton.dissertation.experiments.payloads;

import com.jeffheaton.dissertation.util.NewSimpleEarlyStoppingStrategy;
import com.jeffheaton.dissertation.util.QuickEncodeDataset;
import com.jeffheaton.dissertation.util.Transform;
import org.encog.EncogError;
import org.encog.mathutil.error.ErrorCalculation;
import org.encog.mathutil.error.ErrorCalculationMode;
import org.encog.mathutil.randomize.generate.GenerateRandom;
import org.encog.mathutil.randomize.generate.MersenneTwisterGenerateRandom;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.ea.score.adjust.ComplexityAdjustedScore;
import org.encog.ml.ea.train.basic.TrainEA;
import org.encog.ml.fitness.MultiObjectiveFitness;
import org.encog.ml.prg.EncogProgram;
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
import org.encog.neural.networks.training.TrainingSetScore;
import org.encog.util.Format;
import org.encog.util.Stopwatch;

import java.util.Random;

/**
 * Created by jeff on 6/12/16.
 */
public class PayloadGeneticFit extends AbstractExperimentPayload {

    public static final int POPULATION_SIZE = 100;

    public void verboseStatusGeneticProgram(TrainEA genetic, EncogProgram best, NewSimpleEarlyStoppingStrategy earlyStop, PrgPopulation pop) {
        if( getRunner()==null || getRunner().isVerbose() ) {
            System.out.println(genetic.getIteration() + ", Error: "
                    + Format.formatDouble(best.getScore(), 6) + ",Validation Score: " + earlyStop.getValidationError()
                    + ",Best Genome Size:" + best.size()
                    + ",Species Count:" + pop.getSpecies().size() + ",best: " + best.dumpAsCommonExpression());
        }
    }

    @Override
    public void run() {
        if(!getTask().isRegression()) {
            throw new EncogError("Cannot currently evaluate GP classification.");
        }

        Stopwatch sw = new Stopwatch();
        ErrorCalculation.setMode(ErrorCalculationMode.RMS);
        sw.start();
        // split
        GenerateRandom rnd = new MersenneTwisterGenerateRandom(42);
        org.encog.ml.data.MLDataSet[] split = Transform.splitTrainValidate(getTask().getDataset(), rnd, 0.75);
        MLDataSet trainingSet = split[0];
        MLDataSet validationSet = split[1];

        EncogProgramContext context = new EncogProgramContext();
        for (QuickEncodeDataset.QuickField field: getTask().getQuick().getPredictors()) {
            context.defineVariable(field.getName());
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


        PrgPopulation pop = new PrgPopulation(context, POPULATION_SIZE);

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
        genetic.setThreadCount(1);

        NewSimpleEarlyStoppingStrategy earlyStop = new NewSimpleEarlyStoppingStrategy(validationSet, 5, 50, 0.01);
        genetic.addStrategy(earlyStop);

        (new RampedHalfAndHalf(context, 1, 6)).generate(new Random(), pop);

        genetic.setShouldIgnoreExceptions(false);

        EncogProgram best = null;
        long lastUpdate = System.currentTimeMillis();


        do {
            long sinceLastUpdate = (System.currentTimeMillis() - lastUpdate) / 1000;
            genetic.iteration();
            best = (EncogProgram) genetic.getBestGenome();
            if (getRunner()==null || genetic.getIteration() == 1 || genetic.isTrainingDone() || sinceLastUpdate > 60) {
                verboseStatusGeneticProgram(genetic, best, earlyStop, pop);
            }
        } while (!genetic.isTrainingDone());

        getTask().reportDone((int) (sw.getElapsedMilliseconds() / 1000),earlyStop.getValidationError(),
                genetic.getIteration(),best.dumpAsCommonExpression());
    }
}
