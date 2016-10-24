package com.jeffheaton.dissertation.autofeatures;

import au.com.bytecode.opencsv.CSVWriter;
import org.encog.Encog;
import org.encog.EncogError;
import org.encog.mathutil.error.ErrorCalculation;
import org.encog.mathutil.error.ErrorCalculationMode;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.ea.genome.Genome;
import org.encog.ml.ea.score.adjust.ComplexityAdjustedScore;
import org.encog.ml.ea.train.basic.TrainEA;
import org.encog.ml.prg.EncogProgram;
import org.encog.ml.prg.EncogProgramContext;
import org.encog.ml.prg.PrgCODEC;
import org.encog.ml.prg.ProgramNode;
import org.encog.ml.prg.extension.FunctionFactory;
import org.encog.ml.prg.extension.StandardExtensions;
import org.encog.ml.prg.generator.RampedHalfAndHalf;
import org.encog.ml.prg.opp.ConstMutation;
import org.encog.ml.prg.opp.SubtreeCrossover;
import org.encog.ml.prg.opp.SubtreeMutation;
import org.encog.ml.prg.species.PrgSpeciation;
import org.encog.ml.prg.train.PrgPopulation;
import org.encog.ml.prg.train.rewrite.RewriteConstants;
import org.encog.ml.tree.TreeNode;
import org.encog.util.Format;

import java.io.*;
import java.util.*;

public class AutoEngineerFeatures {
    private int geneticIterations = 1000;
    private File logFeatureDir;
    private MLDataSet trainingSet;
    private MLDataSet validationSet;
    private int populationSize = 100;
    private int hiddenCount = 50;
    private int maxIterations = 50;

    public AutoEngineerFeatures(MLDataSet theTrainingSet, MLDataSet theValidationSet)
    {
        this.trainingSet = theTrainingSet;
        this.validationSet = theValidationSet;
    }

    public void run() {
        ErrorCalculation.setMode(ErrorCalculationMode.RMS);

        EncogProgramContext context = new EncogProgramContext();

        for(int i=1;i<=this.trainingSet.getInputSize();i++) {
            context.defineVariable("x"+i);
        }

        //StandardExtensions.createNumericOperators(context);

        FunctionFactory factory = context.getFunctions();
        factory.addExtension(StandardExtensions.EXTENSION_VAR_SUPPORT);
        factory.addExtension(StandardExtensions.EXTENSION_CONST_SUPPORT);
        factory.addExtension(StandardExtensions.EXTENSION_NEG);
        factory.addExtension(StandardExtensions.EXTENSION_ADD);
        factory.addExtension(StandardExtensions.EXTENSION_SUB);
        factory.addExtension(StandardExtensions.EXTENSION_MUL);
        //factory.addExtension(StandardExtensions.EXTENSION_POWER);
        factory.addExtension(StandardExtensions.EXTENSION_PDIV);


        PrgPopulation pop = new PrgPopulation(context,this.populationSize);

        FeatureScore score = new FeatureScore(this.trainingSet, this.validationSet,pop, this.hiddenCount, this.maxIterations);


        TrainEA genetic = new TrainEA(pop, score);
        //genetic.setValidationMode(true);
        genetic.setCODEC(new PrgCODEC());
        genetic.addOperation(0.5, new SubtreeCrossover());
        genetic.addOperation(0.25, new ConstMutation(context,0.5,1.0));
        genetic.addOperation(0.25, new SubtreeMutation(context,4));
        genetic.addScoreAdjuster(new ComplexityAdjustedScore(5,10,10,500.0));
        pop.getRules().addRewriteRule(new RewriteConstants());
        //genetic.getRules().addRewriteRule(new RewriteAlgebraic());
        genetic.setSpeciation(new PrgSpeciation());
        genetic.setEliteRate(0.5);

        (new RampedHalfAndHalf(context,1, 6)).generate(new Random(), pop);

        genetic.setShouldIgnoreExceptions(true);

        try {
            for (int i = 0; i < geneticIterations; i++) {
                score.calculateScores();
                if( this.logFeatureDir != null ) {
                    dumpFeatures(genetic);
                }
                genetic.iteration();
                System.out.println("Genetic iteration #" + genetic.getIteration() + ", population size: " + pop.size());
            }


        } catch (Throwable t) {
            t.printStackTrace();
        } finally {
            genetic.finishTraining();
            Encog.getInstance().shutdown();
        }
    }

    private void traverse(EncogProgram prg, TreeNode node, Set<String> variables) {
        for(TreeNode childNode : node.getChildNodes()) {
            if( childNode instanceof ProgramNode ) {
                ProgramNode prgNode = (ProgramNode) childNode;
                if (prgNode.getTemplate() == StandardExtensions.EXTENSION_VAR_SUPPORT) {
                    int varIndex = (int)prgNode.getData()[0].toIntValue();
                    String varName = prg.getVariables().getVariableName(varIndex);
                    variables.add(varName);
                }
            }
            traverse(prg,childNode,variables);
        }
    }

    private String findVariables(EncogProgram prg) {
        Set<String> set = new TreeSet<>();
        traverse(prg,prg.getRootNode(),set);
        return set.toString();
    }

    private void calculateStats(EncogProgram prg, double[] stats) {
        double max = Double.NEGATIVE_INFINITY;
        double min = Double.POSITIVE_INFINITY;

        double sum = 0;
        for(MLDataPair pair: this.trainingSet) {
            MLData output = prg.compute(pair.getInput());
            double d = output.getData(0);
            max = Math.max(d,max);
            min = Math.min(d,min);
            sum+=d;
        }
        double mean = sum / this.trainingSet.size();

        sum = 0;
        for(MLDataPair pair: this.trainingSet) {
            MLData output = prg.compute(pair.getInput());
            double d = output.getData(0);
            double diff = mean - d;
            sum+=diff*diff;
        }
        double sdev = Math.sqrt(sum);

        stats[0] = min;
        stats[1] = max;
        stats[2] = mean;
        stats[3] = sdev;

    }

    private void dumpFeatures(TrainEA genetic) {
        CSVWriter writer = null;
        try {
            String filename = "autofeatures-" + genetic.getIteration() + ".csv";
            writer = new CSVWriter(new FileWriter(new File(this.getLogFeatureDir(),filename)));

            writer.writeNext( new String[] {"index","score", "birth","species","vars","min","max","mean","sdev","feature"} );

            List<Genome> list = genetic.getPopulation().flatten();
            int idx = 0;
            double[] stats = new double[4];

            for(Genome genome: list) {
                EncogProgram prg = ((EncogProgram)genome);
                calculateStats(prg, stats);

                String[] line = {
                        ""+idx,
                        Format.formatDouble(genome.getScore(),4),
                        ""+genome.getBirthGeneration(),
                        ""+genetic.getPopulation().getSpecies().indexOf(genome.getSpecies()),
                        findVariables(prg),
                        Format.formatDouble(stats[0],4),
                        Format.formatDouble(stats[1],4),
                        Format.formatDouble(stats[2],4),
                        Format.formatDouble(stats[3],4),
                        prg.dumpAsCommonExpression()
                };
                writer.writeNext(line);
                idx++;
            }

        } catch(IOException ex) {
            throw new EncogError(ex);
        } finally {
            try {
                writer.close();
            } catch (IOException e) {
                throw new EncogError(e);
            }
        }


    }

    public File getLogFeatureDir() {
        return logFeatureDir;
    }

    public void setLogFeatureDir(File logFeatureDir) {
        this.logFeatureDir = logFeatureDir;
    }
}
