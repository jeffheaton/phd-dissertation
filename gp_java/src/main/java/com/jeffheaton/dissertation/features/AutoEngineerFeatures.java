package com.jeffheaton.dissertation.features;

import org.encog.Encog;
import org.encog.mathutil.error.ErrorCalculation;
import org.encog.mathutil.error.ErrorCalculationMode;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.ea.genome.Genome;
import org.encog.ml.ea.score.adjust.ComplexityAdjustedScore;
import org.encog.ml.ea.train.basic.TrainEA;
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

import java.io.*;
import java.util.List;
import java.util.Random;

/**
 * Created by Jeff on 3/31/2016.
 */
public class AutoEngineerFeatures {
    private int geneticIterations = 1000;
    public static File FEATURE_FILE = new File("c:\\data\\features.txt");
    private MLDataSet trainingSet;
    private MLDataSet validationSet;
    private int populationSize = 100;

    public AutoEngineerFeatures(MLDataSet theTrainingSet, MLDataSet theValidationSet)
    {
        this.trainingSet = theTrainingSet;
        this.validationSet = theValidationSet;
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

        //StandardExtensions.createNumericOperators(context);

        FunctionFactory factory = context.getFunctions();
        factory.addExtension(StandardExtensions.EXTENSION_VAR_SUPPORT);
        factory.addExtension(StandardExtensions.EXTENSION_CONST_SUPPORT);
        factory.addExtension(StandardExtensions.EXTENSION_NEG);
        factory.addExtension(StandardExtensions.EXTENSION_ADD);
        factory.addExtension(StandardExtensions.EXTENSION_SUB);
        factory.addExtension(StandardExtensions.EXTENSION_MUL);
        factory.addExtension(StandardExtensions.EXTENSION_PDIV);


        PrgPopulation pop = new PrgPopulation(context,this.populationSize);

        FeatureScore score = new FeatureScore(this.trainingSet, this.validationSet,pop);

        TrainEA genetic = new TrainEA(pop, score);
        //genetic.setValidationMode(true);
        genetic.setCODEC(new PrgCODEC());
        genetic.addOperation(0.5, new SubtreeCrossover());
        genetic.addOperation(0.25, new ConstMutation(context,0.5,1.0));
        genetic.addOperation(0.25, new SubtreeMutation(context,4));
        genetic.addScoreAdjuster(new ComplexityAdjustedScore(5,10,10,500.0));
        genetic.getRules().addRewriteRule(new RewriteConstants());
        genetic.getRules().addRewriteRule(new RewriteAlgebraic());
        genetic.setSpeciation(new PrgSpeciation());

        (new RampedHalfAndHalf(context,1, 6)).generate(new Random(), pop);

        genetic.setShouldIgnoreExceptions(true);

        try {
            for (int i = 0; i < geneticIterations; i++) {
                score.calculateScores();
                dumpFeatures(genetic, FEATURE_FILE);
                genetic.iteration();
                System.out.println(pop.size());
            }


        } catch (Throwable t) {
            t.printStackTrace();
        } finally {
            genetic.finishTraining();
            Encog.getInstance().shutdown();
        }
    }

    public void dumpFeatures(TrainEA genetic, File file) {
        FileOutputStream fos = null;
        try {
            fos = new FileOutputStream(file);
            BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(fos));

            List<Genome> list = genetic.getPopulation().flatten();
            int idx = 0;
            for(Genome genome: list) {
                bw.write( idx + ":" + genome.getScore() + ":" + ((EncogProgram)genome).dumpAsCommonExpression());
                bw.newLine();
                idx++;
            }
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            try {
                fos.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
}
