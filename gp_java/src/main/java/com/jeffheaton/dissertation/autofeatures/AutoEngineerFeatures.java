package com.jeffheaton.dissertation.autofeatures;

import au.com.bytecode.opencsv.CSVWriter;
import org.encog.Encog;
import org.encog.EncogError;
import org.encog.mathutil.error.ErrorCalculation;
import org.encog.mathutil.error.ErrorCalculationMode;
import org.encog.ml.MLMethod;
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
import org.encog.ml.train.BasicTraining;
import org.encog.ml.tree.TreeNode;
import org.encog.neural.networks.training.propagation.TrainingContinuation;
import org.encog.util.Format;

import java.io.*;
import java.util.*;

public class AutoEngineerFeatures extends BasicTraining {
    private MLDataSet trainingSet;
    private MLDataSet validationSet;
    private int populationSize = 100;
    private int hiddenCount = 50;
    private int maxIterations = 5000;
    private TrainEA genetic;
    private FeatureScore score;
    private DumpFeatures dump;

    public AutoEngineerFeatures(MLDataSet theTrainingSet, MLDataSet theValidationSet)
    {
        this.trainingSet = theTrainingSet;
        this.validationSet = theValidationSet;
        this.dump = new DumpFeatures(theTrainingSet);
    }

    private void init() {
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

        this.score = new FeatureScore(this.trainingSet, this.validationSet,pop, this.hiddenCount, this.maxIterations);


        this.genetic = new TrainEA(pop, this.score);
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

    public void iteration() {
        super.preIteration();

        if( this.genetic == null ) {
            init();
        }

        score.calculateScores();

        this.genetic.iteration();
        setError(score.getBestValidationError());
        this.dump.dumpFeatures(getIteration(),this.genetic.getPopulation());

        super.postIteration();
    }

    /**
     * @return True if the training can be paused, and later continued.
     */
    @Override
    public boolean canContinue() {
        return false;
    }

    /**
     * Pause the training to continue later.
     *
     * @return A training continuation object.
     */
    @Override
    public TrainingContinuation pause() {
        return null;
    }

    /**
     * Resume training.
     *
     * @param state The training continuation object to use to continue.
     */
    @Override
    public void resume(TrainingContinuation state) {

    }

    /**
     * Get the current best machine learning method from the training.
     *
     * @return The best machine learning method.
     */
    @Override
    public MLMethod getMethod() {
        return null;
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
}
