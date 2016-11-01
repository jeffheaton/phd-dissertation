package com.jeffheaton.dissertation.experiments.payloads;

import com.jeffheaton.dissertation.JeffDissertation;
import com.jeffheaton.dissertation.experiments.data.ExperimentDatasets;
import com.jeffheaton.dissertation.experiments.manager.ExperimentTask;
import org.encog.mathutil.error.NormalizedError;
import org.encog.mathutil.randomize.generate.GenerateRandom;
import org.encog.mathutil.randomize.generate.MersenneTwisterGenerateRandom;
import org.encog.ml.MLRegression;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.train.MLTrain;
import org.encog.ml.train.strategy.end.EarlyStoppingStrategy;
import org.encog.neural.networks.BasicNetwork;
import org.encog.util.EngineArray;
import org.encog.util.Format;
import org.encog.util.Stopwatch;
import org.encog.util.simple.EncogUtility;

public class PayloadNeuralFit extends AbstractExperimentPayload {

    private BasicNetwork bestNetwork;

    private void statusNeural(ExperimentTask task, MLTrain train, EarlyStoppingStrategy earlyStop) {
        StringBuilder line = new StringBuilder();

        line.append("Epoch #");
        line.append(train.getIteration());
        line.append(" Train Error:");
        line.append(Format.formatDouble(train.getError(), 6));
        line.append(", Validation Error: ");
        line.append(Format.formatDouble(earlyStop.getValidationError(), 6));
        line.append(", Stagnant: ");
        line.append(earlyStop.getStagnantIterations());
        task.log(line.toString());
    }

    @Override
    public PayloadReport run(ExperimentTask task) {
        // get the dataset
        MLDataSet dataset = ExperimentDatasets.getInstance().loadDatasetNeural(
                task.getDatasetFilename(),
                task.getModelType().getTarget(),
                EngineArray.string2list(task.getPredictors())).getData();
        return runWithDataset(task,dataset);

    }

    public PayloadReport runWithDataset(ExperimentTask task, MLDataSet dataset) {
        // split
        GenerateRandom rnd = new MersenneTwisterGenerateRandom(JeffDissertation.RANDOM_SEED);
        org.encog.ml.data.MLDataSet[] split = EncogUtility.splitTrainValidate(dataset, rnd,
                JeffDissertation.TRAIN_VALIDATION_SPLIT);
        MLDataSet trainingSet = split[0];
        MLDataSet validationSet = split[1];

        // Create neural network
        BasicNetwork network = JeffDissertation.factorNeuralNetwork(
                trainingSet.getInputSize(),
                trainingSet.getIdealSize(),
                task.getModelType().isRegression());

        Stopwatch sw = new Stopwatch();
        sw.start();

        // Train neural network
        JeffDissertation.DissertationNeuralTraining d = JeffDissertation.factorNeuralTrainer(
                network,trainingSet,validationSet);
        MLTrain train = d.getTrain();
        EarlyStoppingStrategy earlyStop = d.getEarlyStop();

        long lastUpdate = System.currentTimeMillis();

        do {
            train.iteration();

            long sinceLastUpdate = (System.currentTimeMillis() - lastUpdate) / 1000;

            if (isVerbose() || train.getIteration() == 1 || train.isTrainingDone() || sinceLastUpdate > 60) {
                statusNeural(task, train, earlyStop);
                lastUpdate = System.currentTimeMillis();
            }

            if (Double.isNaN(train.getError()) || Double.isInfinite(train.getError())) {
                break;
            }

        } while (!train.isTrainingDone());
        train.finishTraining();

        MLRegression bestNetwork = earlyStop.getBestModel()==null?network:earlyStop.getBestModel();
        double resultError;

        if( task.getModelType().getError().equalsIgnoreCase("nrmse")) {
            NormalizedError error = new NormalizedError(validationSet);
            resultError = error.calculateNormalizedRange(validationSet, bestNetwork);
        } else {
            resultError = earlyStop.getValidationError();
        }

        sw.stop();

        task.log("Result: " + resultError);
        this.bestNetwork = (BasicNetwork) earlyStop.getBestModel();

        return new PayloadReport(
                (int) (sw.getElapsedMilliseconds() / 1000),
                resultError, earlyStop.getValidationError(), 0, 0,
                train.getIteration(), "");
    }

    public BasicNetwork getBestNetwork() {
        return this.bestNetwork;
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
}
