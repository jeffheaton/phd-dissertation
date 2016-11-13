package com.jeffheaton.dissertation.autofeatures;

import org.encog.EncogError;
import org.encog.StatusReportable;
import org.encog.mathutil.randomize.generate.GenerateRandom;
import org.encog.mathutil.randomize.generate.MersenneTwisterGenerateRandom;
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
import org.encog.neural.networks.training.propagation.sgd.BatchDataSet;
import org.encog.util.concurrency.MultiThreadable;

import java.io.*;
import java.util.*;

public class AutoEngineerFeatures implements MultiThreadable {
    public static final int GP_EPOCS = 50;
    private final List<StatusReportable> listeners = new ArrayList<StatusReportable>();
    private MLDataSet trainingSet;
    private int populationSize = 100;
    private int maxIterations = 500;
    private TrainEA genetic;
    private FeatureScore score;
    private DumpFeatures dump;
    private String[] names;
    private EncogProgramContext context;
    private int threadCount;
    private int maxRankingSet = 10000;
    private MLDataSet rankingSet;
    private GenerateRandom rnd = new MersenneTwisterGenerateRandom();
    private boolean shouldReportNeural = false;


    public AutoEngineerFeatures(MLDataSet theTrainingSet) {
        this.trainingSet = theTrainingSet;
        this.dump = new DumpFeatures(theTrainingSet);
        this.names = new String[this.trainingSet.getInputSize()];
        for (int i = 1; i <= this.trainingSet.getInputSize(); i++) {
            this.names[i - 1] = "x" + i;
        }

    }

    private void sampleRankingSet() {
        Set<Integer> already = new HashSet<Integer>();
        int rankingSize = Math.min(this.maxRankingSet,this.trainingSet.size());

        if( rankingSize<= this.trainingSet.size()) {
            this.rankingSet = this.trainingSet;
        }

        this.rankingSet = new BasicMLDataSet();


        while(this.rankingSet.size()<rankingSize) {
            int idx = this.rnd.nextInt(this.trainingSet.size());
            if( !already.contains(idx) ) {
                already.add(idx);
                MLDataPair pair = this.trainingSet.get(idx);
                this.rankingSet.add(pair);
            }
        }
    }

    private void init() {
        this.context = new EncogProgramContext();
        for (int i = 0; i < this.trainingSet.getInputSize(); i++) {
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
        factory.addExtension(StandardExtensions.EXTENSION_POWER);
        factory.addExtension(StandardExtensions.EXTENSION_PDIV);


        PrgPopulation pop = new PrgPopulation(context, this.populationSize);

        this.score = new FeatureScore(this, this.trainingSet, pop, this.maxIterations);


        this.genetic = new TrainEA(pop, this.score);
        this.genetic.setThreadCount(this.threadCount);
        //genetic.setValidationMode(true);
        this.genetic.setCODEC(new PrgCODEC());
        this.genetic.addOperation(0.5, new SubtreeCrossover());
        this.genetic.addOperation(0.25, new ConstMutation(context, 0.5, 1.0));
        this.genetic.addOperation(0.25, new SubtreeMutation(context, 4));
        this.genetic.addScoreAdjuster(new ComplexityAdjustedScore(5, 10, 10, 500.0));
        pop.getRules().addRewriteRule(new RewriteConstants());
        //genetic.getRules().addRewriteRule(new RewriteAlgebraic());
        this.genetic.setSpeciation(new PrgSpeciation());
        this.genetic.setEliteRate(0.5);

        (new RampedHalfAndHalf(context, 1, 6)).generate(new Random(), pop);

        this.genetic.setShouldIgnoreExceptions(true);

        if( this.rankingSet == null ) {
            sampleRankingSet();
        }
    }

    public void process() {
        init();
        int regenCount = 0;

        for (int i = 0; i < GP_EPOCS; i++) {
            report("Running iteration: " + i);
            this.dump.dumpFeatures(i, this.genetic.getPopulation());
            if( !score.calculateScores() ) {
                regenCount++;

                if( regenCount> FeatureScore.MAX_RETRY_STABLE ) {
                    throw new EncogError("Auto generated failed, could not create stable GP population after " + FeatureScore.MAX_RETRY_STABLE + " tries.");
                }

                report("Population cannot generate a FV stable enough for neural training, resetting entire population.");
                this.genetic.getPopulation().clear();
                (new RampedHalfAndHalf(context, 1, 6)).generate(new Random(), genetic.getPopulation());
            } else {
                this.genetic.iteration();
            }
        }
    }

    public File getLogFeatureDir() {
        return this.dump.getLogFeatureDir();
    }

    public void setLogFeatureDir(File logFeatureDir) {
        this.dump.setLogFeatureDir(logFeatureDir);
    }

    public DumpFeatures getDumpFeatures() {
        return dump;
    }

    public List<EncogProgram> getFeatures(int num) {
        HashSet<String> foundAlready = new HashSet<>();
        List<EncogProgram> result = new ArrayList<>();
        List<Genome> l = this.genetic.getPopulation().flatten();
        Collections.sort(l, this.genetic.getBestComparator());
        int idx = l.size() - 1;

        while (idx >= 0 && result.size() < num) {
            EncogProgram prg = (EncogProgram) l.get(idx);
            String str = prg.dumpAsCommonExpression();
            if (prg.size() > 1 && prg.getScore() > 0 && !foundAlready.contains(str)) {
                result.add(prg);
                foundAlready.add(str);
            }
            idx--;
        }

        return result;
    }

    public void setNames(String[] names) {
        if (names.length != this.names.length) {
            throw new EncogError("Invalid number of field names, expected " + this.names.length + ", but got " + names.length);
        }

        for (int i = 0; i < names.length; i++) {
            this.names[i] = names[i];
        }
    }

    public MLDataSet augmentDataset(int num, MLDataSet dataset) {
        List<EncogProgram> engineeredFeatures = getFeatures(num);
        MLDataSet result = new BasicMLDataSet();
        int inputSize = engineeredFeatures.size() + dataset.getInputSize();

        for (MLDataPair pair : dataset) {
            MLData augmentedInput = new BasicMLData(inputSize);
            MLData augmentedIdeal = new BasicMLData(dataset.getIdealSize());
            MLDataPair augmentedPair = new BasicMLDataPair(augmentedInput, augmentedIdeal);

            // Copy ideal
            for (int i = 0; i < pair.getIdeal().size(); i++) {
                augmentedIdeal.setData(i, pair.getIdeal().getData(i));
            }

            // Create input - Copy origional features
            int idx = 0;
            for (int i = 0; i < pair.getInput().size(); i++) {
                augmentedInput.setData(idx++, pair.getInput().getData(i));
            }

            int i = 0;
            while (idx < inputSize) {
                double d = 0.0;

                if (i < engineeredFeatures.size()) {
                    MLRegression phen = engineeredFeatures.get(i);
                    try {
                        MLData output = phen.compute(pair.getInput());
                        d = output.getData(0);
                    } catch (EARuntimeError ex) {
                        d = 0.0;
                    }
                }
                i++;
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

    public List<StatusReportable> getListeners() {
        return listeners;
    }

    public void addListener(StatusReportable listener) {
        this.listeners.add(listener);
    }

    public void report(String str) {
        for (StatusReportable listener : this.listeners) {
            listener.report(0, 0, str);
        }
    }

    public int getMaxRankingSet() {
        return maxRankingSet;
    }

    public void setMaxRankingSet(int maxRankingSet) {
        this.maxRankingSet = maxRankingSet;
    }

    public MLDataSet getRankingSet() {
        return rankingSet;
    }

    public boolean isShouldReportNeural() {
        return shouldReportNeural;
    }

    public void setShouldReportNeural(boolean shouldReportNeural) {
        this.shouldReportNeural = shouldReportNeural;
    }
}
