package com.jeffheaton.dissertation.experiments.payloads;

import com.jeffheaton.dissertation.experiments.data.DataCacheElement;
import com.jeffheaton.dissertation.experiments.data.ExperimentDatasets;
import com.jeffheaton.dissertation.experiments.manager.ExperimentTask;
import com.jeffheaton.dissertation.features.FindPatternsGP;
import org.encog.EncogError;
import org.encog.mathutil.error.ErrorCalculation;
import org.encog.mathutil.error.ErrorCalculationMode;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.prg.EncogProgram;
import org.encog.util.EngineArray;
import org.encog.util.Stopwatch;

import java.util.List;

public class PayloadPatterns extends AbstractExperimentPayload  {
    public static int N = 100;

    @Override
    public PayloadReport run(ExperimentTask task) {
        DataCacheElement cache = ExperimentDatasets.getInstance().loadDatasetGP(task.getDatasetFilename(),
                task.getModelType().getTarget(),
                EngineArray.string2list(task.getPredictors()));

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
                gp.getNormalizedError(), 0,0,0,
                N, util.reportString(5,N));
    }


}
