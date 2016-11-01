package com.jeffheaton.dissertation.experiments.payloads;

import com.jeffheaton.dissertation.JeffDissertation;
import com.jeffheaton.dissertation.experiments.data.DataCacheElement;
import com.jeffheaton.dissertation.experiments.data.ExperimentDatasets;
import com.jeffheaton.dissertation.experiments.manager.ExperimentTask;
import com.jeffheaton.dissertation.util.*;
import org.encog.EncogError;
import org.encog.mathutil.error.ErrorCalculation;
import org.encog.mathutil.error.ErrorCalculationMode;
import org.encog.mathutil.error.NormalizedError;
import org.encog.mathutil.randomize.generate.GenerateRandom;
import org.encog.mathutil.randomize.generate.MersenneTwisterGenerateRandom;
import org.encog.ml.MLRegression;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.ea.population.Population;
import org.encog.ml.ea.score.adjust.ComplexityAdjustedScore;
import org.encog.ml.ea.train.EvolutionaryAlgorithm;
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
import org.encog.ml.train.MLTrain;
import org.encog.ml.train.strategy.end.EarlyStoppingStrategy;
import org.encog.neural.networks.training.TrainingSetScore;
import org.encog.util.EngineArray;
import org.encog.util.Format;
import org.encog.util.Stopwatch;
import org.encog.util.simple.EncogUtility;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class PayloadGeneticFit extends AbstractExperimentPayload {

    public static String GP_CLASS_ERROR = "GP cannot be used with multiple outputs (classification with more than 2 values)";
    private final List<EncogProgram> best = new ArrayList<>();
    private int n = 1;
    private double rawError;
    private double accumulatedError;
    private int accumulatedRuns;
    private int totalIterations;

    private void verboseStatusGeneticProgram(int current, ExperimentTask task, TrainEA genetic, EncogProgram best, EarlyStoppingStrategy earlyStop, PrgPopulation pop) {
        StringBuilder line = new StringBuilder();

        if( this.n!=1 ) {
            line.append(current);
            line.append("/");
            line.append(this.n);
            line.append(":");
        }
        line.append("Epoch #");
        line.append(genetic.getIteration());
        line.append(" Train Error:");
        line.append(Format.formatDouble(genetic.getError(), 6));
        line.append(", Validation Error: ");
        line.append(Format.formatDouble(earlyStop.getValidationError(), 6));
        line.append(",Best Genome Size:");
        line.append(best.size());
        line.append(",Species Count:");
        line.append(pop.getSpecies().size());
        line.append(", Stagnant: ");
        line.append(earlyStop.getStagnantIterations());
        line.append(",best: ");
        line.append(best.dumpAsCommonExpression());
        task.log(line.toString());
    }

    @Override
    public PayloadReport run(ExperimentTask task) {
        DataCacheElement cache = ExperimentDatasets.getInstance().loadDatasetGP(task.getDatasetFilename(),task.getModelType().getTarget(),
                EngineArray.string2list(task.getPredictors()));
        QuickEncodeDataset quick = cache.getQuick();
        MLDataSet dataset = cache.getData();

        if(dataset.getIdealSize()>2) {
            throw new EncogError(PayloadGeneticFit.GP_CLASS_ERROR);
        }

        // split
        GenerateRandom rnd = new MersenneTwisterGenerateRandom(42);
        org.encog.ml.data.MLDataSet[] split = EncogUtility.splitTrainValidate(dataset, rnd,
                JeffDissertation.TRAIN_VALIDATION_SPLIT);
        MLDataSet trainingSet = split[0];
        MLDataSet validationSet = split[1];

        EncogProgramContext context = JeffDissertation.factorGeneticContext(quick.getFieldNames());

        Stopwatch sw = new Stopwatch();
        ErrorCalculation.setMode(ErrorCalculationMode.RMS);
        sw.start();

        this.totalIterations = 0;
        this.rawError = this.accumulatedError = 0;
        this.accumulatedRuns = 0;
        this.best.clear();

        for(int i=0;i<this.n;i++) {
            fitOne(i+1, task, context,trainingSet,validationSet);
        }

        sw.stop();

        return new PayloadReport(
                (int) (sw.getElapsedMilliseconds() / 1000),getNormalizedError(),getRawError(), 0,0,
                this.totalIterations,this.best.get(0).dumpAsCommonExpression());
    }

    /**
     * Not needed for this payload.
     * @param task Not used.
     * @return Not used.
     */
    @Override
    public MLDataSet obtainCommonProcessing(ExperimentTask task) {
        return null;
    }

    private void fitOne(int current, ExperimentTask task, EncogProgramContext context, MLDataSet trainingSet, MLDataSet validationSet) {

        JeffDissertation.DissertationGeneticTraining d = JeffDissertation.factorGeneticProgramming(context,trainingSet,validationSet,JeffDissertation.POPULATION_SIZE);
        PrgPopulation pop = d.getPopulation();
        EarlyStoppingStrategy earlyStop = d.getEarlyStop();
        TrainEA genetic = d.getTrain();

        long lastUpdate = System.currentTimeMillis();
        int populationFails = 0;


        do {
            long sinceLastUpdate = (System.currentTimeMillis() - lastUpdate) / 1000;

            if (pop.getSpecies().size() == 0) {
                populationFails++;

                (new RampedHalfAndHalf(context, 1, 6)).generate(new Random(), pop);
                if (populationFails >= 5) {
                    throw new EncogError("Entire population invalid after 5 regenerations: ");
                }
            }

            genetic.iteration();
            EncogProgram currentBest = (EncogProgram) genetic.getBestGenome();
            if (isVerbose() || genetic.getIteration() == 1 || genetic.isTrainingDone() || sinceLastUpdate > 60) {
                verboseStatusGeneticProgram(current, task, genetic, currentBest, earlyStop, pop);
            }
        } while (!genetic.isTrainingDone());
        genetic.finishTraining();

        this.totalIterations += genetic.getIteration();
        double resultError;

        if( task.getModelType().getError().equalsIgnoreCase("nrmse")) {
            NormalizedError error = new NormalizedError(validationSet);
            resultError = error.calculateNormalizedMean(validationSet,(MLRegression) genetic.getBestGenome());
        } else {
            resultError = earlyStop.getValidationError();
        }

        if( !Double.isNaN(resultError) && !Double.isInfinite(resultError)) {
            this.accumulatedError += resultError;
            this.accumulatedRuns += 1;
            this.rawError += genetic.getError();
        }
        EncogProgram prg = (EncogProgram) genetic.getBestGenome();
        //prg.setPopulation(null);
        this.best.add(prg);
        pop.clear();

    }

    public List<EncogProgram> getBest() {
        return this.best;
    }

    public int getN() {
        return n;
    }

    public void setN(int n) {
        this.n = n;
    }

    public double getNormalizedError() {
        return this.accumulatedError/this.accumulatedRuns;
    }

    public double getRawError() {
        return this.rawError/this.accumulatedRuns;
    }
}
