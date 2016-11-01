package com.jeffheaton.dissertation.experiments.payloads;

import au.com.bytecode.opencsv.CSVWriter;
import com.jeffheaton.dissertation.JeffDissertation;
import com.jeffheaton.dissertation.autofeatures.AutoEngineerFeatures;
import com.jeffheaton.dissertation.autofeatures.Transform;
import com.jeffheaton.dissertation.experiments.data.DataCacheElement;
import com.jeffheaton.dissertation.experiments.data.ExperimentDatasets;
import com.jeffheaton.dissertation.experiments.manager.DissertationConfig;
import com.jeffheaton.dissertation.experiments.manager.ExperimentTask;
import com.jeffheaton.dissertation.util.QuickEncodeDataset;
import org.encog.Encog;
import org.encog.EncogError;
import org.encog.StatusReportable;
import org.encog.mathutil.error.NormalizedError;
import org.encog.mathutil.randomize.generate.GenerateRandom;
import org.encog.mathutil.randomize.generate.MersenneTwisterGenerateRandom;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.importance.FeatureImportance;
import org.encog.ml.importance.FeatureRank;
import org.encog.ml.importance.PerturbationFeatureImportanceCalc;
import org.encog.neural.networks.BasicNetwork;
import org.encog.util.EngineArray;
import org.encog.util.Stopwatch;
import org.encog.util.csv.CSVFormat;
import org.encog.util.simple.EncogUtility;

import java.io.*;

public class PayloadAutoFeature extends AbstractExperimentPayload implements StatusReportable {

    private ExperimentTask currentTask;

    @Override
    public MLDataSet obtainCommonProcessing(ExperimentTask task) {
        DataCacheElement cache = ExperimentDatasets.getInstance().loadDatasetNeural(task.getDatasetFilename(),task.getModelType().getTarget(),
                EngineArray.string2list(task.getPredictors()));
        QuickEncodeDataset quick = cache.getQuick();
        MLDataSet dataset = cache.getData();

        // split - we will not use the validation set to engineer features
        GenerateRandom rnd = new MersenneTwisterGenerateRandom(JeffDissertation.RANDOM_SEED);
        org.encog.ml.data.MLDataSet[] split = EncogUtility.splitTrainValidate(dataset, rnd,
                JeffDissertation.TRAIN_VALIDATION_SPLIT);
        MLDataSet trainingSet = split[0];

        AutoEngineerFeatures engineer = new AutoEngineerFeatures(trainingSet);
        engineer.addListener(this);

        engineer.setNames(quick.getFieldNames());
        engineer.getDumpFeatures().setLogFeatureDir(DissertationConfig.getInstance().getProjectPath());
        engineer.setLogFeatureDir(DissertationConfig.getInstance().getProjectPath());
        engineer.process();

        String base = new File(task.getDatasetFilename()).getName();
        int k = base.indexOf('.');
        if( k!=-1 ) {
            base = base.substring(0,k);
        }

        // Capture the augmented dataset.
        MLDataSet augmentedSet = engineer.augmentDataset(5, cache.getData());
        File filename = new File(DissertationConfig.getInstance().getPath(task.getName()),"augmented-"+base+".csv");
        try (CSVWriter writer = new CSVWriter(new FileWriter(filename));) {
            int idx = 0;
            String[] headers = generateNames(cache.getData(),augmentedSet,true);

            writer.writeNext(headers);
            for (MLDataPair pair: augmentedSet) {
                String[] line = new String[augmentedSet.getInputSize()+augmentedSet.getIdealSize()];
                int idx2 = 0;
                for(int i=0;i<augmentedSet.getInputSize();i++) {
                    line[idx2++] = CSVFormat.EG_FORMAT.format(pair.getInput().getData(i), Encog.DEFAULT_PRECISION);
                }
                for(int i=0;i<augmentedSet.getIdealSize();i++) {
                    line[idx2++] = CSVFormat.EG_FORMAT.format(pair.getIdeal().getData(i), Encog.DEFAULT_PRECISION);
                }
                writer.writeNext(line);
            }

        } catch (IOException e) {
            e.printStackTrace();
        }

        Transform.zscore(augmentedSet);
        return augmentedSet;
    }

    private String[] generateNames(MLDataSet original, MLDataSet augmented, boolean includeY) {
        int augmentCount = augmented.getInputSize() - original.getInputSize();

        int headerCount = includeY ?
                augmented.getInputSize()+augmented.getIdealSize() :
                augmented.getInputSize();

        String[] headers = new String[headerCount];

        int idx = 0;
        // input (x)
        for(int i=0;i<original.getInputSize();i++) {
            headers[idx++] = "orig-" + i;
        }
        // engineered (x)
        for(int i=0;i<augmentCount;i++) {
            headers[idx++] = "fe-" + i;
        }

        if( includeY) {
            // expected (y)
            for (int i = 0; i < augmented.getIdealSize(); i++) {
                headers[idx++] = "y-" + i;
            }
        }

        return headers;
    }

    @Override
    public PayloadReport run(ExperimentTask task) {

        this.currentTask = task;

        Stopwatch sw = new Stopwatch();
        sw.start();

        DataCacheElement cache = ExperimentDatasets.getInstance().loadDatasetNeural(task.getDatasetFilename(),task.getModelType().getTarget(),
                EngineArray.string2list(task.getPredictors()));
        MLDataSet augmentedDataset = cache.obtainCommonProcessing(task,this);

        PayloadNeuralFit neuralPayload = new PayloadNeuralFit();
        neuralPayload.setVerbose(isVerbose());
        PayloadReport neuralFit = neuralPayload.runWithDataset(task,augmentedDataset);
        sw.stop();

        BasicNetwork network = (BasicNetwork) neuralPayload.getBestNetwork();
        task.log("Best score : " + neuralFit.getResult());
        task.log("Best score (raw): " + neuralFit.getResultRaw());

        task.log("Feature importance (permutation)");
        FeatureImportance fi = new PerturbationFeatureImportanceCalc(); //new NeuralFeatureImportanceCalc();
        String[] names = generateNames(cache.getData(),augmentedDataset,false);
        fi.init(network,names);
        fi.performRanking(augmentedDataset);

        for (FeatureRank ranking : fi.getFeaturesSorted()) {
            task.log(ranking.toString());
        }
        task.log(fi.toString());

        return new PayloadReport(
                (int) (sw.getElapsedMilliseconds() / 1000),
                neuralFit.getResult(), neuralFit.getResultRaw(), 0, 0,
                neuralFit.getIteration(), "");
    }

    /**
     * Report on current status.
     *
     * @param total   The total amount of units to process.
     * @param current The current unit being processed.
     * @param message The status message.
     */
    @Override
    public void report(int total, int current, String message) {
        this.currentTask.log("AutoFeature:" + current + "/" + total + ": " + message);
    }
}
