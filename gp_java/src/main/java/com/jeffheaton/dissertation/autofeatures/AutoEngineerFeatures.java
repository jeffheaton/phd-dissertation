package com.jeffheaton.dissertation.autofeatures;

import org.encog.EncogError;
import org.encog.ml.MLMethod;
import org.encog.ml.MLRegression;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataPair;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.ml.ea.exception.EARuntimeError;
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
import org.encog.ml.prg.train.rewrite.RewriteConstants;
import org.encog.ml.train.BasicTraining;
import org.encog.ml.tree.TreeNode;
import org.encog.neural.networks.training.propagation.TrainingContinuation;
import org.encog.util.Format;
import org.encog.util.concurrency.MultiThreadable;

import java.io.*;
import java.util.*;

public class AutoEngineerFeatures implements MultiThreadable {
    public static final int GP_EPOCS = 50;

    private MLDataSet trainingSet;
    private int populationSize = 100;
    private int hiddenCount = 50;
    private int maxIterations = 500;
    private TrainEA genetic;
    private FeatureScore score;
    private DumpFeatures dump;
    private String[] names;
    private EncogProgramContext context;
    private int threadCount;

    public AutoEngineerFeatures(MLDataSet theTrainingSet)
    {
        this.trainingSet = theTrainingSet;
        this.dump = new DumpFeatures(theTrainingSet);
        this.names = new String[this.trainingSet.getInputSize()];
        for(int i=1;i<=this.trainingSet.getInputSize();i++) {
            this.names[i-1] = "x"+i;
        }

    }

    private void init() {
        this.context = new EncogProgramContext();
        for(int i=0;i<this.trainingSet.getInputSize();i++) {
            context.defineVariable(names[i]);
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

        this.score = new FeatureScore(this, this.trainingSet, pop, this.hiddenCount, this.maxIterations);


        this.genetic = new TrainEA(pop, this.score);
        this.genetic.setThreadCount(this.threadCount);
        //genetic.setValidationMode(true);
        this.genetic.setCODEC(new PrgCODEC());
        this.genetic.addOperation(0.5, new SubtreeCrossover());
        this.genetic.addOperation(0.25, new ConstMutation(context,0.5,1.0));
        this.genetic.addOperation(0.25, new SubtreeMutation(context,4));
        this.genetic.addScoreAdjuster(new ComplexityAdjustedScore(5,10,10,500.0));
        pop.getRules().addRewriteRule(new RewriteConstants());
        //genetic.getRules().addRewriteRule(new RewriteAlgebraic());
        this.genetic.setSpeciation(new PrgSpeciation());
        this.genetic.setEliteRate(0.5);

        (new RampedHalfAndHalf(context,1, 6)).generate(new Random(), pop);

        this.genetic.setShouldIgnoreExceptions(true);

    }

    public void process() {
        init();

        for(int i=0;i<GP_EPOCS;i++) {
            this.dump.dumpFeatures(i, this.genetic.getPopulation());
            score.calculateScores();
            this.genetic.iteration();
        }
    }


    public void setLogFeatureDir(File logFeatureDir) {
        this.dump.setLogFeatureDir(logFeatureDir);
    }

    public File getLogFeatureDir() {
        return this.dump.getLogFeatureDir();
    }

    public DumpFeatures getDumpFeatures() {
        return dump;
    }

    public List<EncogProgram> getFeatures(int num) {
        HashSet<String> foundAlready = new HashSet<>();
        List<EncogProgram> result = new ArrayList<>();
        List<Genome> l = this.genetic.getPopulation().flatten();
        l.sort(this.genetic.getBestComparator());
        int idx = l.size()-1;

        while(idx>=0 && result.size()<num) {
            EncogProgram prg = (EncogProgram)l.get(idx);
            String str = prg.dumpAsCommonExpression();
            if( prg.size()>1 && prg.getScore()>0 && !foundAlready.contains(str) ) {
                result.add(prg);
                foundAlready.add(str);
            }
            idx--;
        }

        return result;
    }

    public void setNames(String[] names) {
        if( names.length != this.names.length) {
            throw new EncogError("Invalid number of field names, expected " + this.names.length + ", but got " + names.length);
        }

        for(int i=0;i<names.length;i++) {
            this.names[i] = names[i];
        }
    }

    public MLDataSet augmentDataset(int num, MLDataSet dataset) {
        List<EncogProgram> engineeredFeatures = getFeatures(num);
        MLDataSet result = new BasicMLDataSet();
        int inputSize = engineeredFeatures.size() + dataset.getInputSize();

        for(MLDataPair pair: dataset) {
            MLData augmentedInput = new BasicMLData(inputSize);
            MLData augmentedIdeal = new BasicMLData(dataset.getIdealSize());
            MLDataPair augmentedPair = new BasicMLDataPair(augmentedInput,augmentedIdeal);

            // Copy ideal
            for(int i=0;i<pair.getIdeal().size();i++){
                augmentedIdeal.setData(i, pair.getIdeal().getData(i));
            }

            // Create input - Copy origional features
            int idx = 0;
            for(int i=0;i<pair.getInput().size();i++) {
                augmentedInput.setData(idx++,pair.getInput().getData(i));
            }

            int i = 0;
            while(idx<inputSize) {
                double d = 0.0;

                if( i< engineeredFeatures.size() ) {
                    MLRegression phen = engineeredFeatures.get(i);
                    try {
                        MLData output = phen.compute(pair.getInput());
                        d = output.getData(0);
                    } catch (EARuntimeError ex) {
                        d = 0.0;
                    }
                }
                augmentedInput.setData(idx++, d);
            }

            FeatureScore.cleanVector(augmentedPair.getInput());
            result.add(augmentedPair);
        }

        return result;
    }

    @Override
    public int getThreadCount() {
        return this.threadCount;
    }

    @Override
    public void setThreadCount(int numThreads) {
        this.threadCount = numThreads;
    }
}
