package com.jeffheaton.dissertation.experiments.payloads;

import org.encog.EncogError;
import org.encog.mathutil.error.ErrorCalculation;
import org.encog.mathutil.error.ErrorCalculationMode;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataPair;
import org.encog.ml.data.basic.BasicMLDataSet;
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
    public PayloadReport run(String[] fields, MLDataSet dataset, boolean regression) {

        if(!regression) {
            throw new EncogError("Cannot currently evaluate GP classification.");
        }

        Stopwatch sw = new Stopwatch();
        ErrorCalculation.setMode(ErrorCalculationMode.RMS);
        sw.start();

        PayloadGeneticFit gp = new PayloadGeneticFit();
        gp.setVerbose(isVerbose());
        gp.setN(N);
        gp.run(fields,dataset,regression);
        List<EncogProgram> gpFeatures = gp.getBest();

        if(isVerbose()) {
            System.out.println("Features:");
            for(EncogProgram prg: gpFeatures) {
                System.out.println(prg.dumpAsCommonExpression());
            }
        }

        // generate ensemble training data
        int originalFeatureCount = dataset.getInputSize();
        int totalFeatureCount = originalFeatureCount + N;
        int totalOutputCount = dataset.getIdealSize();
        MLDataSet ensembleRun = new BasicMLDataSet();
        for(MLDataPair item: dataset) {
            MLData x = new BasicMLData(totalFeatureCount);
            MLData y = new BasicMLData(totalOutputCount);

            int idx = 0;
            for(int i=0;i<originalFeatureCount;i++) {
                x.setData(idx++,item.getInput().getData(i));
            }
            for(int i=0;i<N;i++) {
                MLData output = gpFeatures.get(i).compute(item.getInput());
                x.setData(idx++,output.getData(0));
            }

            for(int i=0;i<totalOutputCount;i++) {
                y.setData(i,item.getIdeal().getData(0));
            }

            MLDataPair newPair = new BasicMLDataPair(x,y);
            ensembleRun.add(newPair);
        }

        // Train neural network
        PayloadNeuralFit neuralPayload = new PayloadNeuralFit();
        neuralPayload.setVerbose(isVerbose());
        PayloadReport neuralFit = neuralPayload.run(null,ensembleRun,regression);
        sw.stop();


        return new PayloadReport(
                (int) (sw.getElapsedMilliseconds() / 1000),
                neuralFit.getResult(),
                neuralFit.getIteration(), "");
    }
}
