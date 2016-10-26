package com.jeffheaton.dissertation.experiments.misc;

import com.jeffheaton.dissertation.experiments.manager.DissertationConfig;
import com.jeffheaton.dissertation.util.SimpleGPConstraint;
import org.encog.Encog;
import org.encog.mathutil.error.ErrorCalculation;
import org.encog.mathutil.error.ErrorCalculationMode;
import org.encog.mathutil.randomize.generate.GenerateRandom;
import org.encog.mathutil.randomize.generate.MersenneTwisterGenerateRandom;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.ml.ea.score.adjust.ComplexityAdjustedScore;
import org.encog.ml.ea.species.SingleSpeciation;
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
import org.encog.neural.networks.training.propagation.sgd.BatchDataSet;
import org.encog.parse.expression.latex.RenderLatexExpression;
import org.encog.util.Format;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class ExperimentExpressions {
    public static final int COUNT = 1000;
    public static final double HIGH = 1;
    public static final double LOW = -1;
    private EncogProgramContext context;
    private MLDataSet trainingSet;
    private MLDataSet validationSet;
    private int inputCount;
    private GenerateRandom rnd = new MersenneTwisterGenerateRandom();


    private void defineContext() {
        this.context = new EncogProgramContext();
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

    private void generateDataset(String expression) {
        EncogProgram prg = new EncogProgram(context,expression);
        System.out.println(prg.toString());
        this.inputCount = prg.getInputCount();
        this.trainingSet = new BasicMLDataSet();
        this.validationSet = new BasicMLDataSet();
        int validationSize = (int)(((double)COUNT)*0.2);

        // Training data
        for(int row=0;row<COUNT;row++) {
            MLData input = new BasicMLData(inputCount);
            for(int col = 0;col<inputCount;col++) {
                input.setData(col,this.rnd.nextDouble(ExperimentExpressions.LOW,ExperimentExpressions.HIGH));
            }
            MLData ideal = prg.compute(input);
            this.trainingSet.add(input,ideal);
        }

        // Validation data
        for(int row=0;row<validationSize;row++) {
            MLData input = new BasicMLData(inputCount);
            for(int col = 0;col<inputCount;col++) {
                input.setData(col,this.rnd.nextDouble(ExperimentExpressions.LOW,ExperimentExpressions.HIGH));
            }
            MLData ideal = prg.compute(input);
            this.validationSet.add(input,ideal);
        }
    }

    private List<String> train(int speciesCount) {

        for (int idx=0; idx< this.inputCount; idx++ ) {
            char ch = (char)('a'+idx);
            context.defineVariable(""+ch);
        }

        List<String> results = new ArrayList<String>();

        PrgPopulation pop = new PrgPopulation(context,100);

        MultiObjectiveFitness score = new MultiObjectiveFitness();
        score.addObjective(1.0, new TrainingSetScore(this.trainingSet));

        TrainEA genetic = new TrainEA(pop, score);
        //genetic.setValidationMode(true);

        genetic.setCODEC(new PrgCODEC());
        genetic.addOperation(0.5, new SubtreeCrossover());
        genetic.addOperation(0.1, new ConstMutation(context,0.5,1.0));
        genetic.addOperation(0.4, new SubtreeMutation(context,3));
        //genetic.addOperation(0.75, new SubtreeMutation(context,5));
        genetic.addScoreAdjuster(new ComplexityAdjustedScore(10,20,10,50.0));
        pop.getRules().addRewriteRule(new RewriteConstants());
        pop.getRules().addRewriteRule(new RewriteAlgebraic());
        pop.getRules().addConstraintRule(new SimpleGPConstraint());

        if(speciesCount!=1) {
            PrgSpeciation sp = new PrgSpeciation();
            sp.setMaxNumberOfSpecies(speciesCount);
            genetic.setSpeciation(sp);
        } else {
            genetic.setSpeciation(new SingleSpeciation());
        }

        EarlyStoppingStrategy earlyStop = new EarlyStoppingStrategy(this.validationSet, 5, 500);
        genetic.addStrategy(earlyStop);

        (new RampedHalfAndHalf(context,1, 6)).generate(new Random(), pop);

        genetic.setShouldIgnoreExceptions(false);

        EncogProgram best = null;
        String lastBest = "";

        try {

            do {
                genetic.iteration();
                best = (EncogProgram) genetic.getBestGenome();
                String bestStr = best.dumpAsCommonExpression();
                String current = (genetic.getIteration() + ", Error: "
                        + Format.formatDouble(best.getScore(),6) + ",Validation Score: " + earlyStop.getValidationError()
                        + ",Best Genome Size:" +best.size()
                        + ",Species Count:" + pop.getSpecies().size() + ",best: " + bestStr);

                if( !lastBest.equals(bestStr)) {
                    lastBest = bestStr;
                    System.out.println(current);
                    results.add(current);
                }
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


            if( best.getScore()<1.0) {
                appendResults(results);
            }

        } catch (Throwable t) {
            t.printStackTrace();
        } finally {
            genetic.finishTraining();
            Encog.getInstance().shutdown();
        }



        return results;
    }

    public void appendResults(List<String> results) {
        try {
            String path = DissertationConfig.getInstance().getProjectPath().toString();
            PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(
                    new File(path,"expressions.txt"), true)));
            for(String line: results) {
                out.println(line);
            }
            out.close();
        } catch (IOException e) {
            //exception handling left as an exercise for the reader
        }
    }

    public void experiment(String expression, int speciesCount) {
        defineContext();
        generateDataset(expression);
        train(speciesCount);
    }

    public static void experimentLoop() {
        ErrorCalculation.setMode(ErrorCalculationMode.RMS);

        for(;;) {
            System.out.println("****");
            ExperimentExpressions prg = new ExperimentExpressions();
            prg.experiment("(a-b)/(c-d)", 30);
        }
    }

    public static void experimentSingle(String expression, int speciesCount) {
        ExperimentExpressions prg = new ExperimentExpressions();
        prg.experiment(expression, speciesCount);
    }

    public static void main(String[] args) {
        ErrorCalculation.setMode(ErrorCalculationMode.RMS);
        //ExperimentExpressions.experimentLoop();
        ExperimentExpressions.experimentSingle("(a-b)/(c-d)", 30);
    }

}
