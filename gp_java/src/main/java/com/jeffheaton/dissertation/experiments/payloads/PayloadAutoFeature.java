package com.jeffheaton.dissertation.experiments.payloads;

import com.jeffheaton.dissertation.autofeatures.AutoEngineerFeatures;
import com.jeffheaton.dissertation.experiments.data.ExperimentDatasets;
import com.jeffheaton.dissertation.experiments.manager.DissertationConfig;
import com.jeffheaton.dissertation.experiments.manager.ExperimentTask;
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

        // get the dataset
        MLDataSet dataset = ExperimentDatasets.getInstance().loadDatasetNeural(
                task.getDatasetFilename(),
                task.getModelType().getTarget(),
                EngineArray.string2list(task.getPredictors())).getData();

        // split
        GenerateRandom rnd = new MersenneTwisterGenerateRandom(42);
        org.encog.ml.data.MLDataSet[] split = EncogUtility.splitTrainValidate(dataset, rnd, 0.75);
        MLDataSet trainingSet = split[0];
        MLDataSet validationSet = split[1];

        AutoEngineerFeatures auto = new AutoEngineerFeatures(trainingSet, validationSet);
        auto.setLogFeatureDir(DissertationConfig.getInstance().getProjectPath());

        for(int i=0;i<50;i++) {
            auto.iteration();
        }

        sw.stop();

        double resultError = auto.getError();
        double resultValidation = 0;
        int steps = 1;

        task.log("Result: " + resultError);

        return new PayloadReport(
                (int) (sw.getElapsedMilliseconds() / 1000),
                resultError, resultValidation, 0, 0,
                steps, "");
    }
}
