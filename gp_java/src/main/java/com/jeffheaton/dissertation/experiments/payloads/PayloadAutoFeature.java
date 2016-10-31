package com.jeffheaton.dissertation.experiments.payloads;

import com.jeffheaton.dissertation.JeffDissertation;
import com.jeffheaton.dissertation.autofeatures.AutoEngineerFeatures;
import com.jeffheaton.dissertation.experiments.data.DataCacheElement;
import com.jeffheaton.dissertation.experiments.data.ExperimentDatasets;
import com.jeffheaton.dissertation.experiments.manager.DissertationConfig;
import com.jeffheaton.dissertation.experiments.manager.ExperimentTask;
import com.jeffheaton.dissertation.util.QuickEncodeDataset;
import org.encog.mathutil.randomize.generate.GenerateRandom;
import org.encog.mathutil.randomize.generate.MersenneTwisterGenerateRandom;
import org.encog.ml.data.MLDataSet;
import org.encog.util.EngineArray;
import org.encog.util.Stopwatch;
import org.encog.util.simple.EncogUtility;

public class PayloadAutoFeature extends AbstractExperimentPayload {

    @Override
    public MLDataSet obtainCommonProcessing(ExperimentTask task) {
        DataCacheElement cache = ExperimentDatasets.getInstance().loadDatasetNeural(task.getDatasetFilename(),task.getModelType().getTarget(),
                EngineArray.string2list(task.getPredictors()));
        QuickEncodeDataset quick = cache.getQuick();
        MLDataSet dataset = cache.getData();

        // split
        GenerateRandom rnd = new MersenneTwisterGenerateRandom(JeffDissertation.RANDOM_SEED);
        org.encog.ml.data.MLDataSet[] split = EncogUtility.splitTrainValidate(dataset, rnd,
                JeffDissertation.TRAIN_VALIDATION_SPLIT);
        MLDataSet trainingSet = split[0];
        MLDataSet validationSet = split[1];

        AutoEngineerFeatures engineer = new AutoEngineerFeatures(trainingSet);

        engineer.setNames(quick.getFieldNames());
        engineer.getDumpFeatures().setLogFeatureDir(DissertationConfig.getInstance().getProjectPath());
        engineer.setLogFeatureDir(DissertationConfig.getInstance().getProjectPath());
        engineer.process();
        return engineer.augmentDataset(5, cache.getData());
    }

    @Override
    public PayloadReport run(ExperimentTask task) {
        Stopwatch sw = new Stopwatch();
        sw.start();

        DataCacheElement cache = ExperimentDatasets.getInstance().loadDatasetNeural(task.getDatasetFilename(),task.getModelType().getTarget(),
                EngineArray.string2list(task.getPredictors()));
        MLDataSet augmentedDataset = cache.obtainCommonProcessing(task,this);

        PayloadNeuralFit neuralPayload = new PayloadNeuralFit();
        neuralPayload.setVerbose(isVerbose());
        PayloadReport neuralFit = neuralPayload.run(task);
        sw.stop();

        return new PayloadReport(
                (int) (sw.getElapsedMilliseconds() / 1000),
                neuralFit.getResult(), neuralFit.getResultRaw(), 0, 0,
                neuralFit.getIteration(), "");
    }
}
