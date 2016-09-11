package com.jeffheaton.dissertation.features.ex2_gp_fit_many;

import com.jeffheaton.dissertation.JeffDissertation;
import com.jeffheaton.dissertation.experiments.manager.DissertationConfig;
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
import org.encog.persist.source.ObtainInputStream;
import org.encog.persist.source.ObtainResourceInputStream;
import org.encog.util.Format;
import org.encog.util.csv.CSVFormat;
import org.encog.util.simple.EncogUtility;

import java.io.InputStream;
import java.util.Random;

/**
 * Created by jeff on 4/16/16.
 */
public class FitManyGP {

    private boolean verboseGP = false;
    private MLDataSet trainingSet;
    private MLDataSet validationSet;

    public static void main(String[] args) {
        InputStream is;
        FitManyGP file = new FitManyGP();
        file.process();
    }

    private void loadData() {
        ObtainInputStream source = new ObtainResourceInputStream(
                "/auto-mpg.csv", JeffDissertation.class);
        QuickEncodeDataset quick = new QuickEncodeDataset(false,false);
        quick.analyze(source,"mpg", true, CSVFormat.EG_FORMAT);
        MLDataSet dataset = quick.generateDataset();

        // split
        MLDataSet[] split = EncogUtility.splitTrainValidate(dataset,new MersenneTwisterGenerateRandom(42),0.75);
        this.trainingSet = split[0];
        this.validationSet = split[1];

        EncogProgramContext context = new EncogProgramContext();
        /*for(int i=0;i<quick.getName().length;i++) {
            if( i!=quick.getTargetIndex() && quick.getNumeric()[i] ) {
                context.defineVariable(quick.getName()[i]);
            }
        }*/
    }

    private void defineOperators(EncogProgramContext context) {
        FunctionFactory factory = context.getFunctions();
        factory.addExtension(StandardExtensions.EXTENSION_VAR_SUPPORT);
        factory.addExtension(StandardExtensions.EXTENSION_CONST_SUPPORT);
        factory.addExtension(StandardExtensions.EXTENSION_NEG);
        factory.addExtension(StandardExtensions.EXTENSION_ADD);
        factory.addExtension(StandardExtensions.EXTENSION_SUB);
        factory.addExtension(StandardExtensions.EXTENSION_MUL);
        factory.addExtension(StandardExtensions.EXTENSION_DIV);
        factory.addExtension(StandardExtensions.EXTENSION_POWER);
    }

    public PrgPopulation generatePopulation(EncogProgramContext context) {
        PrgPopulation pop = new PrgPopulation(context,1000);
        (new RampedHalfAndHalf(context,1, 6)).generate(new Random(), pop);
        return pop;
    }

    private TrainEA createTrainer(PrgPopulation pop) {
        EncogProgramContext context = pop.getContext();
        ErrorCalculation.setMode(ErrorCalculationMode.RMS);

        MultiObjectiveFitness score = new MultiObjectiveFitness();
        score.addObjective(1.0, new TrainingSetScore(trainingSet));

        TrainEA genetic = new TrainEA(pop, score);
        genetic.setCODEC(new PrgCODEC());
        genetic.addOperation(0.5, new SubtreeCrossover());
        genetic.addOperation(0.25, new ConstMutation(context,0.5,1.0));
        genetic.addOperation(0.25, new SubtreeMutation(context,4));
        genetic.addScoreAdjuster(new ComplexityAdjustedScore(10,20,10,50.0));
        pop.getRules().addRewriteRule(new RewriteConstants());
        pop.getRules().addRewriteRule(new RewriteAlgebraic());
        genetic.setSpeciation(new PrgSpeciation());
        genetic.setShouldIgnoreExceptions(false);
        return genetic;
    }

    private EarlyStoppingStrategy defineEarlyStop(TrainEA train) {
        EarlyStoppingStrategy earlyStop = new EarlyStoppingStrategy(validationSet, 5, 500, 0.01);
        train.addStrategy(earlyStop);
        return earlyStop;
    }



    private void process() {
        loadData();
        EncogProgramContext context = new EncogProgramContext();
        defineOperators(context);
        PrgPopulation pop = generatePopulation(context);
        TrainEA train = createTrainer(pop);
        EarlyStoppingStrategy earlyStop = defineEarlyStop(train);

        EncogProgram best = null;

        try {

            do {
                train.iteration();
                best = (EncogProgram) train.getBestGenome();
                System.out.println(train.getIteration() + ", Error: "
                        + Format.formatDouble(best.getScore(),6) + ",Validation Score: " + earlyStop.getValidationError()
                        + ",Best Genome Size:" +best.size()
                        + ",Species Count:" + pop.getSpecies().size() + ",best: " + best.dumpAsCommonExpression());
            } while(!train.isTrainingDone());

            //EncogUtility.evaluate(best, trainingData);

            System.out.println("Final score:" + best.getScore()
                    + ", effective score:" + best.getAdjustedScore());
            System.out.println(best.dumpAsCommonExpression());
            System.out.println();
            //pop.dumpMembers(Integer.MAX_VALUE);
            //pop.dumpMembers(10);

            RenderLatexExpression latex = new RenderLatexExpression();
            EncogProgram bestProgram = (EncogProgram) train.getBestGenome();
            String str = latex.render(bestProgram);
            System.out.println("Latex: " + str);

        } catch (Throwable t) {
            t.printStackTrace();
        } finally {
            train.finishTraining();
            Encog.getInstance().shutdown();
        }


    }
}
