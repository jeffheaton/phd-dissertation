package com.jeffheaton.dissertation.experiments.payloads;

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

    public static final int MINI_BATCH_SIZE = 50;
    public static final double LEARNING_RATE = 1e-2;
    public static final int STAGNANT_NEURAL = 50;
    public static final double L1 = 0;
    public static final double L2 = 1e-8;

    @Override
    public PayloadReport run(ExperimentTask task) {
        Stopwatch sw = new Stopwatch();
        sw.start();

        DataCacheElement cache = ExperimentDatasets.getInstance().loadDatasetNeural(task.getDatasetFilename(),task.getModelType().getTarget(),
                EngineArray.string2list(task.getPredictors()));
        QuickEncodeDataset quick = cache.getQuick();
        MLDataSet dataset = cache.getData();

        // split
        GenerateRandom rnd = new MersenneTwisterGenerateRandom(42);
        org.encog.ml.data.MLDataSet[] split = EncogUtility.splitTrainValidate(dataset, rnd, 0.75);
        MLDataSet trainingSet = split[0];
        MLDataSet validationSet = split[1];

        AutoEngineerFeatures engineer = new AutoEngineerFeatures(trainingSet, validationSet);
        engineer.setNames(quick.getFieldNames());
        engineer.getDumpFeatures().setLogFeatureDir(DissertationConfig.getInstance().getProjectPath());
        engineer.setLogFeatureDir(DissertationConfig.getInstance().getProjectPath());
        engineer.process();

        double resultError = 0;
        double resultValidation = 0;
        int steps = 1;

        task.log("Result: " + resultError);

        return new PayloadReport(
                (int) (sw.getElapsedMilliseconds() / 1000),
                resultError, resultValidation, 0, 0,
                steps, "");
    }
}
