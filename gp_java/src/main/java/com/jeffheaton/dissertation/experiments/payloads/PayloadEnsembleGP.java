package com.jeffheaton.dissertation.experiments.payloads;

import com.jeffheaton.dissertation.experiments.data.DataCacheElement;
import com.jeffheaton.dissertation.experiments.data.ExperimentDatasets;
import com.jeffheaton.dissertation.experiments.manager.ExperimentTask;
import com.jeffheaton.dissertation.util.ArrayUtils;
import com.jeffheaton.dissertation.util.QuickEncodeDataset;
import org.encog.EncogError;
import org.encog.mathutil.error.ErrorCalculation;
import org.encog.mathutil.error.ErrorCalculationMode;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataPair;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.ml.ea.exception.EARuntimeError;
import org.encog.ml.prg.EncogProgram;
import org.encog.util.Stopwatch;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by jeff on 6/15/16.
 */
public class PayloadEnsembleGP extends AbstractExperimentPayload {

    public static int N = 5;

    @Override
    public PayloadReport run(ExperimentTask task) {

        Stopwatch sw = new Stopwatch();
        ErrorCalculation.setMode(ErrorCalculationMode.RMS);
        sw.start();

        DataCacheElement cacheNeural = ExperimentDatasets.getInstance().loadDatasetNeural(task.getDatasetFilename(),task.getModelType().getTarget(), ArrayUtils.string2list(task.getPredictors()));
        DataCacheElement cacheGP = ExperimentDatasets.getInstance().loadDatasetGP(task.getDatasetFilename(),task.getModelType().getTarget(), ArrayUtils.string2list(task.getPredictors()));

        QuickEncodeDataset quickNeural = cacheNeural.getQuick();
        QuickEncodeDataset quickGP = cacheGP.getQuick();

        MLDataSet datasetNeural = quickNeural.generateDataset();
        MLDataSet datasetGP = quickGP.generateDataset();

        if(datasetGP.getIdealSize()>1) {
            throw new EncogError(PayloadGeneticFit.GP_CLASS_ERROR);
        }

        PayloadGeneticFit gp = new PayloadGeneticFit();
        gp.setVerbose(isVerbose());
        gp.setN(N);
        gp.run(task);
        List<EncogProgram> gpFeatures = gp.getBest();

        if(isVerbose()) {
            System.out.println("Features:");
            for(EncogProgram prg: gpFeatures) {
                System.out.println(prg.dumpAsCommonExpression());
            }
        }

        // generate ensemble training data
        int originalFeatureCount = datasetNeural.getInputSize();
        int totalFeatureCount = originalFeatureCount + N;
        int totalOutputCount = datasetGP.getIdealSize();
        MLDataSet ensembleRun = new BasicMLDataSet();

        for(int itemnum=0;itemnum<datasetGP.size();itemnum++) {
            MLDataPair itemNeural = datasetNeural.get(itemnum);
            MLDataPair itemGP = datasetGP.get(itemnum);

            MLData x = new BasicMLData(totalFeatureCount);
            MLData y = new BasicMLData(totalOutputCount);

            int idx = 0;
            for(int i=0;i<originalFeatureCount;i++) {
                x.setData(idx++,itemNeural.getInput().getData(i));
            }
            for(int i=0;i<N;i++) {
                try {
                    MLData output = gpFeatures.get(i).compute(itemGP.getInput());
                    x.setData(idx++, output.getData(0));
                } catch(EARuntimeError ex) {
                    // division by zero, usually.
                    x.setData(idx++,0);
                }
            }

            for(int i=0;i<totalOutputCount;i++) {
                y.setData(i,itemGP.getIdeal().getData(0));
            }

            MLDataPair newPair = new BasicMLDataPair(x,y);
            ensembleRun.add(newPair);
        }

        // Train neural network
        PayloadNeuralFit neuralPayload = new PayloadNeuralFit();
        neuralPayload.setVerbose(isVerbose());
        PayloadReport neuralFit = neuralPayload.run(task);
        sw.stop();


        return new PayloadReport(
                (int) (sw.getElapsedMilliseconds() / 1000),
                neuralFit.getResult(), neuralFit.getResultRaw(), neuralFit.getI1(), neuralFit.getI2(),
                neuralFit.getIteration(), "");
    }
}
