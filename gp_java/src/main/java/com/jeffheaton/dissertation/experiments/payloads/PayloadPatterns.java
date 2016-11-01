package com.jeffheaton.dissertation.experiments.payloads;

import com.jeffheaton.dissertation.JeffDissertation;
import com.jeffheaton.dissertation.experiments.data.DataCacheElement;
import com.jeffheaton.dissertation.experiments.data.ExperimentDatasets;
import com.jeffheaton.dissertation.experiments.manager.ExperimentTask;
import com.jeffheaton.dissertation.util.FindPatternsGP;
import org.encog.EncogError;
import org.encog.mathutil.error.ErrorCalculation;
import org.encog.mathutil.error.ErrorCalculationMode;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.prg.EncogProgram;
import org.encog.util.EngineArray;
import org.encog.util.Stopwatch;

import java.util.List;

public class PayloadPatterns extends AbstractExperimentPayload  {

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
        gp.setN(JeffDissertation.EXP4_PATTERN_COUNT);
        gp.run(task);
        List<EncogProgram> gpFeatures = gp.getBest();

        FindPatternsGP util = new FindPatternsGP();
        for(EncogProgram prg: gpFeatures ) {
            util.find(prg);
        }

        return new PayloadReport(
                (int) (sw.getElapsedMilliseconds() / 1000),
                gp.getNormalizedError(), 0,0,0,
                JeffDissertation.EXP4_PATTERN_COUNT,
                util.reportString(5,JeffDissertation.EXP4_PATTERN_COUNT));
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
