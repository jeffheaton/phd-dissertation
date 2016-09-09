package com.jeffheaton.dissertation.experiments.misc;

import com.jeffheaton.dissertation.util.*;
import org.encog.Encog;
import org.encog.mathutil.error.ErrorCalculation;
import org.encog.mathutil.error.ErrorCalculationMode;
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
import org.encog.ml.train.strategy.end.EarlyStoppingStrategy;
import org.encog.neural.networks.training.TrainingSetScore;
import org.encog.parse.expression.latex.RenderLatexExpression;
import org.encog.util.Format;
import org.encog.util.csv.CSVFormat;
import org.encog.util.simple.EncogUtility;

import java.io.InputStream;
import java.util.Random;

/**
 * Created by jeff on 4/11/16.
 */
public class ExperimentGPFile {
    public static void main(String[] args) {
        InputStream is;
        ExperimentGPFile file = new ExperimentGPFile();
        file.process();
    }

    private void process() {
        ErrorCalculation.setMode(ErrorCalculationMode.RMS);

        //String filename = "auto-mpg.csv";
        //String target = "mpg";
        //String predictors = null;

        //String filename = "feature_eng.csv";
        //String target = "ratio-y0";
        //String predictors = "ratio-x0,ratio-x1";

        String filename = "feature_eng.csv";
        String target = "ratio_diff-y0";
        String predictors = "ratio_diff-x0,ratio_diff-x1,ratio_diff-x2,ratio_diff-x3";

        ObtainInputStream source = new ObtainFallbackStream(filename);
        QuickEncodeDataset quick = new QuickEncodeDataset(true,false);
        quick.analyze(source,target, true, CSVFormat.EG_FORMAT);
        quick.forcePredictors(predictors);
        MLDataSet dataset = quick.generateDataset();

        // split
        MLDataSet[] split = EncogUtility.splitTrainValidate(dataset,new MersenneTwisterGenerateRandom(42),0.75);
        MLDataSet trainingSet = split[0];
        MLDataSet validationSet = split[1];

        EncogProgramContext context = new EncogProgramContext();
        int idx = 0;
        for (String field: quick.getFieldNames()) {
            char ch = (char)('a'+idx);
            idx++;
            context.defineVariable(""+ch);
        }

        //StandardExtensions.createNumericOperators(context);

        FunctionFactory factory = context.getFunctions();
        factory.addExtension(StandardExtensions.EXTENSION_VAR_SUPPORT);
        factory.addExtension(StandardExtensions.EXTENSION_CONST_SUPPORT);
        factory.addExtension(StandardExtensions.EXTENSION_NEG);
        factory.addExtension(StandardExtensions.EXTENSION_ADD);
        factory.addExtension(StandardExtensions.EXTENSION_SUB);
        factory.addExtension(StandardExtensions.EXTENSION_MUL);
        factory.addExtension(StandardExtensions.EXTENSION_PDIV);
        factory.addExtension(StandardExtensions.EXTENSION_POWER);


        PrgPopulation pop = new PrgPopulation(context,100);

        MultiObjectiveFitness score = new MultiObjectiveFitness();
        score.addObjective(1.0, new TrainingSetScore(trainingSet));

        TrainEA genetic = new TrainEA(pop, score);
        //genetic.setValidationMode(true);
        genetic.setCODEC(new PrgCODEC());
        genetic.addOperation(0.5, new SubtreeCrossover());
        genetic.addOperation(0.25, new ConstMutation(context,0.5,1.0));
        genetic.addOperation(0.25, new SubtreeMutation(context,4));
        //genetic.addOperation(0.75, new SubtreeMutation(context,5));
        genetic.addScoreAdjuster(new ComplexityAdjustedScore(10,20,10,50.0));
        pop.getRules().addRewriteRule(new RewriteConstants());
        pop.getRules().addRewriteRule(new RewriteAlgebraic());
        pop.getRules().addConstraintRule(new SimpleGPConstraint());
        genetic.setSpeciation(new PrgSpeciation());

        EarlyStoppingStrategy earlyStop = new EarlyStoppingStrategy(validationSet, 5, 500, 0.01);
        genetic.addStrategy(earlyStop);

        (new RampedHalfAndHalf(context,1, 6)).generate(new Random(), pop);

        EncogProgram prg = new EncogProgram("a/(c-d)");
        prg.setPopulation(pop);
        System.out.println("Error: " + EncogUtility.calculateRegressionError(prg,validationSet));
        //pop.getSpecies().get(0).add(prg);
        //pop.dumpMembers(100);

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
            System.out.println();
            //pop.dumpMembers(Integer.MAX_VALUE);
            //pop.dumpMembers(10);

            RenderLatexExpression latex = new RenderLatexExpression();
            EncogProgram bestProgram = (EncogProgram) genetic.getBestGenome();
            String str = latex.render(bestProgram);
            System.out.println("Latex: " + str);

        } catch (Throwable t) {
            t.printStackTrace();
        } finally {
            genetic.finishTraining();
            Encog.getInstance().shutdown();
        }


    }
}
