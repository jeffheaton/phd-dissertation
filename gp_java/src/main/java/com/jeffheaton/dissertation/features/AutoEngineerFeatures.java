package com.jeffheaton.dissertation.features;

import com.jeffheaton.dissertation.util.Transform;
import org.encog.Encog;
import org.encog.mathutil.error.ErrorCalculation;
import org.encog.mathutil.error.ErrorCalculationMode;
import org.encog.ml.data.MLDataSet;
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
import org.encog.neural.networks.training.TrainingSetScore;
import org.encog.util.csv.CSVFormat;
import org.encog.util.csv.ReadCSV;

import java.io.InputStream;
import java.util.Random;

/**
 * Created by Jeff on 3/31/2016.
 */
public class AutoEngineerFeatures {
    private MLDataSet dataset;

    public AutoEngineerFeatures(MLDataSet theDataset) {
        this.dataset = theDataset;
    }

    public void run() {
        ErrorCalculation.setMode(ErrorCalculationMode.RMS);

        EncogProgramContext context = new EncogProgramContext();
        context.defineVariable("c");
        context.defineVariable("d");
        context.defineVariable("h");
        context.defineVariable("w");
        context.defineVariable("a");
        context.defineVariable("y");
        context.defineVariable("o");

        StandardExtensions.createNumericOperators(context);

        PrgPopulation pop = new PrgPopulation(context,100);

        FeatureScore score = new FeatureScore(this.dataset,pop);

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

        (new RampedHalfAndHalf(context,1, 6)).generate(new Random(), pop);

        genetic.setShouldIgnoreExceptions(false);

        try {
            for (int i = 0; i < 1000; i++) {
                score.calculateScores();
                genetic.iteration();
                System.out.println(genetic.getIteration());
            }

            //EncogUtility.evaluate(best, trainingData);
        } catch (Throwable t) {
            t.printStackTrace();
        } finally {
            genetic.finishTraining();
            Encog.getInstance().shutdown();
        }
    }
}
