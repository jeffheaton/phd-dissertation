package com.jeffheaton.dissertation.experiments.payloads;

import com.jeffheaton.dissertation.experiments.data.DataCacheElement;
import com.jeffheaton.dissertation.experiments.data.ExperimentDatasets;
import com.jeffheaton.dissertation.experiments.manager.ExperimentTask;
import com.jeffheaton.dissertation.features.FindPatternsGP;
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
import org.encog.ml.prg.EncogProgram;
import org.encog.util.Stopwatch;

import java.util.List;

/**
 * Created by jeff on 6/25/16.
 */
public class PayloadPatterns extends AbstractExperimentPayload  {
    public static int N = 500;

    @Override
    public PayloadReport run(ExperimentTask task) {
        DataCacheElement cache = ExperimentDatasets.getInstance().loadDatasetNeural(task);
        QuickEncodeDataset quick = cache.getQuick();
        MLDataSet dataset = cache.getData();

        if(dataset.getIdealSize()>2) {
            throw new EncogError(PayloadGeneticFit.GP_CLASS_ERROR);
        }

        Stopwatch sw = new Stopwatch();
        ErrorCalculation.setMode(ErrorCalculationMode.RMS);
        sw.start();

        PayloadGeneticFit gp = new PayloadGeneticFit();
        gp.setVerbose(isVerbose());
        gp.setN(N);
        gp.run(task);
        List<EncogProgram> gpFeatures = gp.getBest();

        FindPatternsGP util = new FindPatternsGP();
        for(EncogProgram prg: gpFeatures ) {
            util.find(prg);
        }

        return new PayloadReport(
                (int) (sw.getElapsedMilliseconds() / 1000),
                -1,
                N, util.reportString(5,N));
    }


}
